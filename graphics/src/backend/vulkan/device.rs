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

/// The queues to create on the logical device.
///
/// Planned before device creation from the physical device's queue families:
/// the graphics queue (always), plus an async compute queue when the hardware
/// exposes one (#47 phase 4). A dedicated compute-capable family is preferred
/// (real overlap on most discrete GPUs); a second queue in the graphics
/// family is the fallback; otherwise there is no async queue and everything
/// runs on the graphics queue — a first-class, fully supported mode
/// (MoltenVK's default configuration, for one, exposes a single queue).
#[derive(Debug, Clone, Copy)]
pub struct QueuePlan {
    /// Queue family of the graphics queue (queue index 0 within it).
    pub graphics_family: u32,
    /// `(family, queue index)` of the async compute queue, if available.
    pub async_compute: Option<(u32, u32)>,
}

/// Plan which queues to create on the device.
pub fn plan_queues(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> Result<QueuePlan, GraphicsError> {
    let queue_families =
        unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

    let graphics_family = queue_families
        .iter()
        .position(|family| family.queue_flags.contains(vk::QueueFlags::GRAPHICS))
        .ok_or_else(|| {
            GraphicsError::InitializationFailed("No graphics queue family found".to_string())
        })? as u32;

    // Prefer a dedicated compute family; fall back to a second queue in the
    // graphics family.
    let dedicated_compute = queue_families.iter().enumerate().find(|(_, family)| {
        family.queue_flags.contains(vk::QueueFlags::COMPUTE)
            && !family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
            && family.queue_count > 0
    });
    let async_compute = match dedicated_compute {
        Some((family, _)) => Some((family as u32, 0)),
        None if queue_families[graphics_family as usize].queue_count > 1 => {
            Some((graphics_family, 1))
        }
        None => None,
    };

    match async_compute {
        Some((family, index)) => log::info!(
            "Async compute queue planned: family {family}, queue {index}{}",
            if family == graphics_family {
                " (graphics family)"
            } else {
                " (dedicated family)"
            }
        ),
        None => log::info!("No async compute queue available; single-queue mode"),
    }

    Ok(QueuePlan {
        graphics_family,
        async_compute,
    })
}

/// Create a logical device with required features and extensions.
pub fn create_logical_device(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    plan: &QueuePlan,
) -> Result<ash::Device, GraphicsError> {
    // Two priority slots so a same-family plan can create two queues from
    // one create-info; a different-family plan uses one slot per info.
    let queue_priorities = [1.0f32, 1.0f32];
    let mut queue_create_infos = vec![
        vk::DeviceQueueCreateInfo::default()
            .queue_family_index(plan.graphics_family)
            .queue_priorities(&queue_priorities[..1]),
    ];
    match plan.async_compute {
        Some((family, _)) if family == plan.graphics_family => {
            queue_create_infos[0] = queue_create_infos[0].queue_priorities(&queue_priorities[..2]);
        }
        Some((family, _)) => {
            queue_create_infos.push(
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(family)
                    .queue_priorities(&queue_priorities[..1]),
            );
        }
        None => {}
    }

    // Required device extensions
    #[allow(unused_mut)]
    let mut device_extensions = vec![
        ash::khr::swapchain::NAME.as_ptr(),
        ash::khr::dynamic_rendering::NAME.as_ptr(),
    ];

    // On macOS with MoltenVK, we need VK_KHR_portability_subset
    #[cfg(target_os = "macos")]
    {
        // Check if portability subset is supported and required
        if device_supports_extension(instance, physical_device, "VK_KHR_portability_subset") {
            device_extensions.push(c"VK_KHR_portability_subset".as_ptr());
        }
    }

    // Enable required features:
    // - sampler_anisotropy: anisotropic texture filtering.
    // - fill_mode_non_solid: wireframe (PolygonMode::Line) pipelines.
    let features = vk::PhysicalDeviceFeatures::default()
        .sampler_anisotropy(true)
        .fill_mode_non_solid(true);

    // Enable dynamic rendering via extension features (works on Vulkan 1.2 with extension)
    // This is compatible with MoltenVK which only supports Vulkan 1.2
    let mut dynamic_rendering_features =
        vk::PhysicalDeviceDynamicRenderingFeatures::default().dynamic_rendering(true);

    // shaderDrawParameters: shaders that read SV_InstanceID compile to SPIR-V
    // using the DrawParameters capability, which requires this feature.
    let mut vulkan11_features =
        vk::PhysicalDeviceVulkan11Features::default().shader_draw_parameters(true);

    // timelineSemaphore: the queue timeline that backs frame fences and (with
    // multi-queue, #47) cross-queue waits. Mandatory on every Vulkan 1.2
    // device (our minimum API version, incl. MoltenVK), but must still be
    // explicitly enabled.
    let mut vulkan12_features =
        vk::PhysicalDeviceVulkan12Features::default().timeline_semaphore(true);

    let create_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_create_infos)
        .enabled_extension_names(&device_extensions)
        .enabled_features(&features)
        .push_next(&mut dynamic_rendering_features)
        .push_next(&mut vulkan11_features)
        .push_next(&mut vulkan12_features);

    let device =
        unsafe { instance.create_device(physical_device, &create_info, None) }.map_err(|e| {
            GraphicsError::InitializationFailed(format!("Failed to create logical device: {:?}", e))
        })?;

    Ok(device)
}

/// Check if a physical device supports a specific extension.
#[cfg(target_os = "macos")]
fn device_supports_extension(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    extension_name: &str,
) -> bool {
    let extensions =
        match unsafe { instance.enumerate_device_extension_properties(physical_device) } {
            Ok(ext) => ext,
            Err(_) => return false,
        };

    for ext in extensions {
        let name = unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) };
        if let Ok(name_str) = name.to_str()
            && name_str == extension_name
        {
            return true;
        }
    }

    false
}
