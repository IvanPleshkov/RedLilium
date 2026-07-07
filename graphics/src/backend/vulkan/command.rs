//! Vulkan command pool and buffer management.

use ash::vk;

use crate::error::GraphicsError;

/// Create a command pool for graphics operations.
///
/// `per_buffer_reset` enables `RESET_COMMAND_BUFFER` — required only for
/// pools whose buffers are reset individually (the shared pool's present
/// command buffers). Pools that are only ever bulk-reset per frame slot must
/// pass `false`: the flag forces drivers into a general-purpose allocator
/// instead of a cheap linear one.
pub fn create_command_pool(
    device: &ash::Device,
    queue_family_index: u32,
    per_buffer_reset: bool,
) -> Result<vk::CommandPool, GraphicsError> {
    let flags = if per_buffer_reset {
        vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER
    } else {
        vk::CommandPoolCreateFlags::empty()
    };
    let pool_info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(queue_family_index)
        .flags(flags);

    let pool = unsafe { device.create_command_pool(&pool_info, None) }.map_err(|e| {
        GraphicsError::InitializationFailed(format!("Failed to create command pool: {:?}", e))
    })?;

    Ok(pool)
}
