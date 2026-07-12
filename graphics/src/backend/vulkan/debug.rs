//! Vulkan validation layer debug messenger.

use std::cell::Cell;
use std::ffi::CStr;

use ash::vk;

use crate::error::GraphicsError;

thread_local! {
    /// Count of `ERROR`-severity `VALIDATION` messages seen on *this* thread.
    ///
    /// The validation layer invokes the debug callback synchronously on the
    /// thread making the offending Vulkan call, so a test that drives all its
    /// GPU work on one thread can bracket a section with
    /// [`reset_validation_error_count`] / [`validation_error_count`] to assert
    /// zero VUIDs without races from other tests' instances.
    static VALIDATION_ERRORS: Cell<usize> = const { Cell::new(0) };
}

/// Number of Vulkan validation errors recorded on the current thread since the
/// last [`reset_validation_error_count`].
pub fn validation_error_count() -> usize {
    VALIDATION_ERRORS.with(|c| c.get())
}

/// Reset the current thread's validation-error counter to zero.
pub fn reset_validation_error_count() {
    VALIDATION_ERRORS.with(|c| c.set(0));
}

/// Create a debug messenger for validation layer output.
pub fn create_debug_messenger(
    debug_utils: &ash::ext::debug_utils::Instance,
) -> Result<vk::DebugUtilsMessengerEXT, GraphicsError> {
    let create_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
        // No INFO: loader/driver info chatter drowns real findings on every
        // validation run (VK-L11); errors and warnings are what we act on.
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
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

    // `message_type` is a bitmask and may combine flags; exact-equality
    // matching mislabeled combined types as "Unknown" (VK-L11).
    let type_str = if message_type.contains(vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION) {
        "Validation"
    } else if message_type.contains(vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE) {
        "Performance"
    } else if message_type.contains(vk::DebugUtilsMessageTypeFlagsEXT::GENERAL) {
        "General"
    } else {
        "Unknown"
    };

    match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => {
            if message_type.contains(vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION) {
                VALIDATION_ERRORS.with(|c| c.set(c.get() + 1));
            }
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
