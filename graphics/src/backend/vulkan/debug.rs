//! Vulkan validation layer debug messenger.

use std::ffi::CStr;

use ash::vk;

use crate::error::GraphicsError;

/// Create a debug messenger for validation layer output.
pub fn create_debug_messenger(
    debug_utils: &ash::ext::debug_utils::Instance,
) -> Result<vk::DebugUtilsMessengerEXT, GraphicsError> {
    let create_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
        )
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
        )
        .pfn_user_callback(Some(debug_callback));

    let messenger = unsafe { debug_utils.create_debug_utils_messenger(&create_info, None) }
        .map_err(|e| {
            GraphicsError::InitializationFailed(format!(
                "Failed to create debug messenger: {:?}",
                e
            ))
        })?;

    Ok(messenger)
}

/// Debug callback function for validation layer messages.
unsafe extern "system" fn debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::ffi::c_void,
) -> vk::Bool32 {
    // SAFETY: This function is only called by the Vulkan driver with valid data
    let message = if callback_data.is_null() {
        String::from("(no message)")
    } else {
        // SAFETY: callback_data is guaranteed to be valid by the Vulkan driver
        let data = unsafe { *callback_data };
        if data.p_message.is_null() {
            String::from("(null message)")
        } else {
            // SAFETY: p_message is a valid null-terminated string from the Vulkan driver
            unsafe { CStr::from_ptr(data.p_message) }
                .to_string_lossy()
                .into_owned()
        }
    };

    let type_str = match message_type {
        vk::DebugUtilsMessageTypeFlagsEXT::GENERAL => "General",
        vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION => "Validation",
        vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE => "Performance",
        _ => "Unknown",
    };

    match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => {
            log::error!("[Vulkan {}] {}", type_str, message);
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => {
            log::warn!("[Vulkan {}] {}", type_str, message);
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => {
            log::info!("[Vulkan {}] {}", type_str, message);
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => {
            log::debug!("[Vulkan {}] {}", type_str, message);
        }
        _ => {
            log::trace!("[Vulkan {}] {}", type_str, message);
        }
    }

    vk::FALSE
}
