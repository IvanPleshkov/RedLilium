//! Vulkan physical and logical device management.

use std::ffi::CStr;

use ash::vk;

use crate::error::GraphicsError;

/// Select the best physical device for rendering.
///
/// Prefers discrete GPUs over integrated GPUs.
pub fn select_physical_device(
    instance: &ash::Instance,
) -> Result<vk::PhysicalDevice, GraphicsError> {
    let devices = unsafe { instance.enumerate_physical_devices() }.map_err(|e| {
        GraphicsError::InitializationFailed(format!(
            "Failed to enumerate physical devices: {:?}",
            e
        ))
    })?;

    if devices.is_empty() {
        return Err(GraphicsError::InitializationFailed(
            "No Vulkan-capable GPU found".to_string(),
        ));
    }

    // Score and select best device
    let mut best_device = None;
    let mut best_score = 0;

    for device in devices {
        let properties = unsafe { instance.get_physical_device_properties(device) };
        let features = unsafe { instance.get_physical_device_features(device) };

        // Check for required features
        if features.sampler_anisotropy == vk::FALSE {
            continue;
        }

        // Score the device
        let mut score = 0;

        // Prefer discrete GPUs
        if properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
            score += 1000;
        } else if properties.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU {
            score += 100;
        }

        // Add score based on max texture size
        score += properties.limits.max_image_dimension2_d / 1024;

        if score > best_score {
            best_score = score;
            best_device = Some(device);
        }

        // Log device info
        let device_name = unsafe { CStr::from_ptr(properties.device_name.as_ptr()) };
        log::info!(
            "Found GPU: {:?} (type: {:?}, score: {})",
            device_name,
            properties.device_type,
            score
        );
    }

    best_device
        .ok_or_else(|| GraphicsError::InitializationFailed("No suitable GPU found".to_string()))
}

/// Find a queue family that supports graphics operations.
pub fn find_graphics_queue_family(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> Result<u32, GraphicsError> {
    let queue_families =
        unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

    for (index, family) in queue_families.iter().enumerate() {
        if family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
            return Ok(index as u32);
        }
    }

    Err(GraphicsError::InitializationFailed(
        "No graphics queue family found".to_string(),
    ))
}

/// Create a logical device with required features and extensions.
pub fn create_logical_device(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    graphics_queue_family: u32,
) -> Result<ash::Device, GraphicsError> {
    let queue_priorities = [1.0f32];
    let queue_create_info = vk::DeviceQueueCreateInfo::default()
        .queue_family_index(graphics_queue_family)
        .queue_priorities(&queue_priorities);

    let queue_create_infos = [queue_create_info];

    // Required device extensions
    let device_extensions = [
        ash::khr::swapchain::NAME.as_ptr(),
        ash::khr::dynamic_rendering::NAME.as_ptr(),
    ];

    // Enable required features
    let features = vk::PhysicalDeviceFeatures::default().sampler_anisotropy(true);

    // Enable Vulkan 1.3 features for dynamic rendering
    let mut vulkan_13_features =
        vk::PhysicalDeviceVulkan13Features::default().dynamic_rendering(true);

    // Enable synchronization2 for better barrier API (optional but recommended)
    vulkan_13_features = vulkan_13_features.synchronization2(true);

    let create_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_create_infos)
        .enabled_extension_names(&device_extensions)
        .enabled_features(&features)
        .push_next(&mut vulkan_13_features);

    let device =
        unsafe { instance.create_device(physical_device, &create_info, None) }.map_err(|e| {
            GraphicsError::InitializationFailed(format!("Failed to create logical device: {:?}", e))
        })?;

    Ok(device)
}
