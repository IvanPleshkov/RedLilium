//! Vulkan instance creation and configuration.

use std::collections::HashSet;
use std::ffi::{CStr, CString};

use ash::vk;

use crate::error::GraphicsError;

use super::debug;

/// The highest Vulkan API version the engine targets.
///
/// The instance requests `min(PREFERRED, loader version)` — a 1.2-only loader
/// (e.g. MoltenVK) gets 1.2 instead of failing (VK-L10). Anything below
/// [`MINIMUM_API_VERSION`] is rejected: the baseline tier (ADR-027) needs
/// Vulkan 1.2 core features (timeline semaphores).
const PREFERRED_API_VERSION: u32 = vk::make_api_version(0, 1, 3, 0);

/// The lowest Vulkan API version the engine can run on.
const MINIMUM_API_VERSION: u32 = vk::make_api_version(0, 1, 2, 0);

/// Validation layer name.
const VALIDATION_LAYER_NAME: &CStr = c"VK_LAYER_KHRONOS_validation";

/// A freshly created Vulkan instance plus what was actually enabled on it.
pub struct CreatedInstance {
    pub instance: ash::Instance,
    pub debug_messenger: Option<vk::DebugUtilsMessengerEXT>,
    pub debug_utils: Option<ash::ext::debug_utils::Instance>,
    /// Whether `VK_KHR_surface` (and at least one platform surface extension)
    /// was available and enabled. False means a headless loader (e.g. CI
    /// without a display): offscreen rendering works, surface creation and
    /// the swapchain device extension are unavailable.
    pub surface_support: bool,
    /// Spec version of `VK_LAYER_KHRONOS_validation` when it was enabled
    /// (`None` = validation off or unavailable). Lets device creation skip
    /// extensions the layer predates — an unknown extension makes the layer
    /// emit false-positive errors and disable its handling of that extension.
    pub validation_layer_spec: Option<u32>,
}

/// Create a Vulkan instance with optional validation layers.
///
/// Extensions are enabled only after checking
/// `enumerate_instance_extension_properties` (VK-M11): a Wayland-only or
/// headless system gets a working instance with whatever surface path exists
/// instead of `vkCreateInstance` failing on a hardcoded extension list.
pub fn create_instance(
    entry: &ash::Entry,
    validation_enabled: bool,
) -> Result<CreatedInstance, GraphicsError> {
    // Negotiate the API version with the loader instead of hardcoding it.
    // `try_enumerate_instance_version` returns None on Vulkan 1.0 loaders.
    let loader_version = unsafe { entry.try_enumerate_instance_version() }
        .map_err(|e| {
            GraphicsError::InitializationFailed(format!(
                "Failed to query Vulkan loader version: {:?}",
                e
            ))
        })?
        .unwrap_or(vk::API_VERSION_1_0);
    if loader_version < MINIMUM_API_VERSION {
        return Err(GraphicsError::InitializationFailed(format!(
            "Vulkan loader is version {}.{}, but the engine requires at least 1.2",
            vk::api_version_major(loader_version),
            vk::api_version_minor(loader_version),
        )));
    }
    let api_version = PREFERRED_API_VERSION.min(loader_version);
    log::info!(
        "Vulkan instance API version {}.{} (loader supports {}.{})",
        vk::api_version_major(api_version),
        vk::api_version_minor(api_version),
        vk::api_version_major(loader_version),
        vk::api_version_minor(loader_version),
    );

    // Check if validation layers are available
    let validation_layer_spec = if validation_enabled {
        validation_layer_spec_version(entry)
    } else {
        None
    };
    let validation_available = validation_layer_spec.is_some();

    if validation_enabled && !validation_available {
        log::warn!("Validation layers requested but not available");
    }

    // What the loader actually offers — every extension below is checked
    // against this set before being enabled.
    let available = available_instance_extensions(entry)?;
    let has = |name: &CStr| available.contains(name);

    // Application info
    let app_name = CString::new("RedLilium").unwrap();
    let engine_name = CString::new("RedLilium Engine").unwrap();

    let app_info = vk::ApplicationInfo::default()
        .application_name(&app_name)
        .application_version(vk::make_api_version(0, 0, 1, 0))
        .engine_name(&engine_name)
        .engine_version(vk::make_api_version(0, 0, 1, 0))
        .api_version(api_version);

    let mut extensions: Vec<*const i8> = Vec::new();

    // Surface support: VK_KHR_surface plus every available platform surface
    // extension (the window's handle type decides which one is used at
    // surface-creation time, so enabling all present ones is correct — a
    // Wayland-only loader simply won't offer xlib/xcb).
    let mut platform_surface = false;
    if has(ash::khr::surface::NAME) {
        let platform_extensions: &[&CStr] = &[
            #[cfg(target_os = "windows")]
            ash::khr::win32_surface::NAME,
            #[cfg(target_os = "linux")]
            ash::khr::xlib_surface::NAME,
            #[cfg(target_os = "linux")]
            ash::khr::xcb_surface::NAME,
            #[cfg(target_os = "linux")]
            ash::khr::wayland_surface::NAME,
            #[cfg(target_os = "macos")]
            ash::ext::metal_surface::NAME,
        ];
        for ext in platform_extensions {
            if has(ext) {
                extensions.push(ext.as_ptr());
                platform_surface = true;
            }
        }
        if platform_surface {
            extensions.push(ash::khr::surface::NAME.as_ptr());
        }
    }
    let surface_support = platform_surface;
    if !surface_support {
        log::info!("No Vulkan surface extensions available; headless mode (offscreen only)");
    }

    // Add debug utils extension if validation is enabled
    if validation_available && has(ash::ext::debug_utils::NAME) {
        extensions.push(ash::ext::debug_utils::NAME.as_ptr());
    }

    // Enabled layers
    let layer_names: Vec<*const i8> = if validation_available {
        vec![VALIDATION_LAYER_NAME.as_ptr()]
    } else {
        vec![]
    };

    // Instance creation flags
    let mut create_flags = vk::InstanceCreateFlags::empty();
    if has(ash::khr::portability_enumeration::NAME) {
        // MoltenVK (and other portability implementations) are only
        // enumerated when explicitly opted into.
        extensions.push(ash::khr::portability_enumeration::NAME.as_ptr());
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
    let (debug_messenger, debug_utils) = if validation_available && has(ash::ext::debug_utils::NAME)
    {
        let debug_utils = ash::ext::debug_utils::Instance::new(entry, &instance);
        let messenger = debug::create_debug_messenger(&debug_utils)?;
        (Some(messenger), Some(debug_utils))
    } else {
        (None, None)
    };

    Ok(CreatedInstance {
        instance,
        debug_messenger,
        debug_utils,
        surface_support,
        validation_layer_spec,
    })
}

/// All instance extensions the loader offers.
fn available_instance_extensions(entry: &ash::Entry) -> Result<HashSet<CString>, GraphicsError> {
    let props = unsafe { entry.enumerate_instance_extension_properties(None) }.map_err(|e| {
        GraphicsError::InitializationFailed(format!(
            "Failed to enumerate instance extensions: {:?}",
            e
        ))
    })?;
    Ok(props
        .iter()
        .map(|p| unsafe { CStr::from_ptr(p.extension_name.as_ptr()) }.to_owned())
        .collect())
}

/// Spec version of the validation layer, or `None` when it is not installed.
fn validation_layer_spec_version(entry: &ash::Entry) -> Option<u32> {
    let available_layers = unsafe { entry.enumerate_instance_layer_properties() }.ok()?;

    available_layers
        .iter()
        .find(|layer| {
            let name = unsafe { CStr::from_ptr(layer.layer_name.as_ptr()) };
            name == VALIDATION_LAYER_NAME
        })
        .map(|layer| layer.spec_version)
}
