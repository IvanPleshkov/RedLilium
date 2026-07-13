//! Hand-rolled `VK_KHR_maintenance9` bindings (#88 step 2).
//!
//! ash 0.38 tracks Vulkan headers 1.3.281, which predate the extension
//! (2025); the structs below mirror `vulkan_core.h` and should be replaced
//! with ash's own definitions once a release ships them.
//!
//! What the extension gives us: queue-family ownership transfers become
//! optional — never needed for buffers and linear images, and a per-queue-
//! family property reports which *other* families may implicitly acquire
//! optimal-tiling images owned by this family without their contents
//! becoming undefined. Where the graphics and async compute families
//! mutually support that, declared cross-queue textures can stay
//! `EXCLUSIVE` (keeping framebuffer compression) with no transfer barriers —
//! the D3D12 model. Synchronization is unchanged: the tracker-emitted
//! timeline waits already order cross-queue access (#47).

use std::ffi::{CStr, c_void};

use ash::vk;

/// `VK_KHR_MAINTENANCE_9_EXTENSION_NAME`
pub const EXTENSION_NAME: &CStr = c"VK_KHR_maintenance9";

/// First Vulkan header release defining the extension (2025-06). A
/// validation layer older than this reports our hand-rolled structs as
/// "unknown VkStructureType" errors and disables its handling of the
/// extension, so maintenance9 is skipped when validating under one.
pub const MIN_AWARE_HEADER_VERSION: u32 = vk::make_api_version(0, 1, 4, 316);

/// `VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_9_FEATURES_KHR`
const STYPE_FEATURES: vk::StructureType = vk::StructureType::from_raw(1000584000);
/// `VK_STRUCTURE_TYPE_QUEUE_FAMILY_OWNERSHIP_TRANSFER_PROPERTIES_KHR`
const STYPE_QUEUE_FAMILY_OWNERSHIP_TRANSFER: vk::StructureType =
    vk::StructureType::from_raw(1000584002);

/// `VkPhysicalDeviceMaintenance9FeaturesKHR`
#[repr(C)]
pub struct PhysicalDeviceMaintenance9FeaturesKHR {
    s_type: vk::StructureType,
    p_next: *mut c_void,
    pub maintenance9: vk::Bool32,
}

impl Default for PhysicalDeviceMaintenance9FeaturesKHR {
    fn default() -> Self {
        Self {
            s_type: STYPE_FEATURES,
            p_next: std::ptr::null_mut(),
            maintenance9: vk::FALSE,
        }
    }
}

impl PhysicalDeviceMaintenance9FeaturesKHR {
    /// Features struct requesting `maintenance9 = TRUE` (device creation).
    pub fn enabled() -> Self {
        Self {
            maintenance9: vk::TRUE,
            ..Self::default()
        }
    }
}

// SAFETY: repr(C) with sType/pNext leading fields, sType matches the
// extension's registered value — a valid member of both pNext chains per the
// Vulkan registry.
unsafe impl vk::ExtendsPhysicalDeviceFeatures2 for PhysicalDeviceMaintenance9FeaturesKHR {}
unsafe impl vk::ExtendsDeviceCreateInfo for PhysicalDeviceMaintenance9FeaturesKHR {}

/// `VkQueueFamilyOwnershipTransferPropertiesKHR`
#[repr(C)]
pub struct QueueFamilyOwnershipTransferPropertiesKHR {
    s_type: vk::StructureType,
    p_next: *mut c_void,
    /// Bitmask of queue family indices that may implicitly acquire
    /// optimal-tiling images owned by THIS queue family without the
    /// contents becoming undefined.
    pub optimal_image_transfer_to_queue_families: u32,
}

impl Default for QueueFamilyOwnershipTransferPropertiesKHR {
    fn default() -> Self {
        Self {
            s_type: STYPE_QUEUE_FAMILY_OWNERSHIP_TRANSFER,
            p_next: std::ptr::null_mut(),
            optimal_image_transfer_to_queue_families: 0,
        }
    }
}

// SAFETY: as above, for the VkQueueFamilyProperties2 output chain.
unsafe impl vk::ExtendsQueueFamilyProperties2 for QueueFamilyOwnershipTransferPropertiesKHR {}

/// Whether the device reports the `maintenance9` feature (call only after
/// confirming the extension is listed — unknown pNext sTypes are skipped by
/// drivers, which would read as `false` here anyway).
pub fn feature_supported(instance: &ash::Instance, device: vk::PhysicalDevice) -> bool {
    let mut m9 = PhysicalDeviceMaintenance9FeaturesKHR::default();
    let mut features2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut m9);
    unsafe { instance.get_physical_device_features2(device, &mut features2) };
    m9.maintenance9 == vk::TRUE
}

/// Whether optimal-tiling images owned by either of the two queue families
/// may be implicitly acquired by the other (both directions — the engine
/// ping-pongs ownership: graphics renders, async compute reads, graphics
/// reuses).
pub fn implicit_image_transfer_ok(
    instance: &ash::Instance,
    device: vk::PhysicalDevice,
    family_a: u32,
    family_b: u32,
) -> bool {
    let count = unsafe { instance.get_physical_device_queue_family_properties2_len(device) } as u32;
    if family_a >= count || family_b >= count || family_a >= 32 || family_b >= 32 {
        return false;
    }

    let mut transfer_props: Vec<QueueFamilyOwnershipTransferPropertiesKHR> =
        (0..count).map(|_| Default::default()).collect();
    let mut family_props: Vec<vk::QueueFamilyProperties2> = transfer_props
        .iter_mut()
        .map(|t| vk::QueueFamilyProperties2::default().push_next(t))
        .collect();
    unsafe { instance.get_physical_device_queue_family_properties2(device, &mut family_props) };
    drop(family_props);

    for (family, props) in transfer_props.iter().enumerate() {
        log::debug!(
            "maintenance9: family {family} optimal images implicitly acquirable by families {:#b}",
            props.optimal_image_transfer_to_queue_families
        );
    }

    let may_acquire_from = |owner: u32, acquirer: u32| {
        transfer_props[owner as usize].optimal_image_transfer_to_queue_families & (1 << acquirer)
            != 0
    };
    may_acquire_from(family_a, family_b) && may_acquire_from(family_b, family_a)
}
