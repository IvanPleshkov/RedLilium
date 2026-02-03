//! Vulkan instance creation and configuration.

use std::ffi::{CStr, CString};

use ash::vk;

use crate::error::GraphicsError;

use super::debug;

/// Required Vulkan API version.
/// On macOS with MoltenVK, only Vulkan 1.2 is supported.
/// On other platforms, we can use 1.3 for native dynamic rendering support.
#[cfg(target_os = "macos")]
const REQUIRED_API_VERSION: u32 = vk::make_api_version(0, 1, 2, 0);

#[cfg(not(target_os = "macos"))]
const REQUIRED_API_VERSION: u32 = vk::make_api_version(0, 1, 3, 0);

/// Validation layer name.
const VALIDATION_LAYER_NAME: &CStr = c"VK_LAYER_KHRONOS_validation";

/// Create a Vulkan instance with optional validation layers.
///
/// Returns the instance, debug messenger (if validation enabled), and debug utils extension.
pub fn create_instance(
    entry: &ash::Entry,
    validation_enabled: bool,
) -> Result<
    (
        ash::Instance,
        Option<vk::DebugUtilsMessengerEXT>,
        Option<ash::ext::debug_utils::Instance>,
    ),
    GraphicsError,
> {
    // Check if validation layers are available
    let validation_available = validation_enabled && check_validation_layer_support(entry);

    if validation_enabled && !validation_available {
        log::warn!("Validation layers requested but not available");
    }

    // Application info
    let app_name = CString::new("RedLilium").unwrap();
    let engine_name = CString::new("RedLilium Engine").unwrap();

    let app_info = vk::ApplicationInfo::default()
        .application_name(&app_name)
        .application_version(vk::make_api_version(0, 0, 1, 0))
        .engine_name(&engine_name)
        .engine_version(vk::make_api_version(0, 0, 1, 0))
        .api_version(REQUIRED_API_VERSION);

    // Required extensions
    let mut extensions = vec![
        ash::khr::surface::NAME.as_ptr(),
        // Platform-specific surface extensions would go here
    ];

    // Add debug utils extension if validation is enabled
    if validation_available {
        extensions.push(ash::ext::debug_utils::NAME.as_ptr());
    }

    // Add platform-specific surface extension
    #[cfg(target_os = "windows")]
    {
        extensions.push(ash::khr::win32_surface::NAME.as_ptr());
    }

    #[cfg(target_os = "linux")]
    {
        extensions.push(ash::khr::xlib_surface::NAME.as_ptr());
        extensions.push(ash::khr::wayland_surface::NAME.as_ptr());
    }

    #[cfg(target_os = "macos")]
    {
        extensions.push(ash::khr::portability_enumeration::NAME.as_ptr());
        extensions.push(ash::ext::metal_surface::NAME.as_ptr());
    }

    // Enabled layers
    let layer_names: Vec<*const i8> = if validation_available {
        vec![VALIDATION_LAYER_NAME.as_ptr()]
    } else {
        vec![]
    };

    // Instance creation flags
    #[allow(unused_mut)]
    let mut create_flags = vk::InstanceCreateFlags::empty();

    #[cfg(target_os = "macos")]
    {
        create_flags |= vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR;
    }

    // Create instance
    let create_info = vk::InstanceCreateInfo::default()
        .flags(create_flags)
        .application_info(&app_info)
        .enabled_extension_names(&extensions)
        .enabled_layer_names(&layer_names);

    let instance = unsafe { entry.create_instance(&create_info, None) }.map_err(|e| {
        GraphicsError::InitializationFailed(format!("Failed to create Vulkan instance: {:?}", e))
    })?;

    // Setup debug messenger if validation is enabled
    let (debug_messenger, debug_utils) = if validation_available {
        let debug_utils = ash::ext::debug_utils::Instance::new(entry, &instance);
        let messenger = debug::create_debug_messenger(&debug_utils)?;
        (Some(messenger), Some(debug_utils))
    } else {
        (None, None)
    };

    Ok((instance, debug_messenger, debug_utils))
}

/// Check if the validation layer is available.
fn check_validation_layer_support(entry: &ash::Entry) -> bool {
    let available_layers = match unsafe { entry.enumerate_instance_layer_properties() } {
        Ok(layers) => layers,
        Err(_) => return false,
    };

    for layer in &available_layers {
        let name = unsafe { CStr::from_ptr(layer.layer_name.as_ptr()) };
        if name == VALIDATION_LAYER_NAME {
            return true;
        }
    }

    false
}
