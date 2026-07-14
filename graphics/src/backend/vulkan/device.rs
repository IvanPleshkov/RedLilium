//! Vulkan physical and logical device management.
//!
//! Selection is filter-then-score (ADR-027): devices that cannot run the
//! baseline tier are excluded with a logged reason *before* scoring, so the
//! highest-scored device is always one that `vkCreateDevice` will accept —
//! the previous "score first, fail later" shape could pick a device that
//! failed creation while a lesser device would have worked (VK-L8).

use std::collections::HashSet;
use std::ffi::CStr;

use ash::vk;

use crate::error::GraphicsError;

/// Features the engine uses when present, degrading gracefully when absent
/// (ADR-027 capabilities, not tier requirements).
#[derive(Debug, Clone, Copy)]
pub struct OptionalFeatures {
    /// Anisotropic filtering; without it samplers are created isotropic.
    pub sampler_anisotropy: bool,
    /// Wireframe (`PolygonMode::Line`) pipelines; without it line mode
    /// downgrades to fill.
    pub fill_mode_non_solid: bool,
    /// `VK_KHR_maintenance9` (#88): queue-family ownership transfers become
    /// optional where the driver says so — declared cross-queue textures can
    /// then stay EXCLUSIVE (keeping compression) instead of CONCURRENT.
    pub maintenance9: bool,
}

/// The physical device chosen by [`select_physical_device`] plus everything
/// queried about it during selection.
pub struct SelectedDevice {
    pub physical_device: vk::PhysicalDevice,
    pub properties: vk::PhysicalDeviceProperties,
    pub optional: OptionalFeatures,
    /// Whether the device lists `VK_KHR_portability_subset` (it must then be
    /// enabled at logical-device creation, per spec).
    pub portability_subset: bool,
}

/// Why a physical device cannot run the baseline tier.
///
/// Returned per-device during filtering; every named gap is logged so a
/// selection failure on exotic hardware is diagnosable from the log alone.
fn baseline_gaps(
    instance: &ash::Instance,
    device: vk::PhysicalDevice,
    require_swapchain: bool,
    extensions: &HashSet<Vec<u8>>,
) -> Vec<&'static str> {
    let mut gaps = Vec::new();

    let properties = unsafe { instance.get_physical_device_properties(device) };
    if properties.api_version < vk::make_api_version(0, 1, 2, 0) {
        gaps.push("Vulkan 1.2");
    }

    let queue_families = unsafe { instance.get_physical_device_queue_family_properties(device) };
    if !queue_families
        .iter()
        .any(|f| f.queue_flags.contains(vk::QueueFlags::GRAPHICS))
    {
        gaps.push("graphics queue family");
    }

    let has_ext = |name: &CStr| extensions.contains(name.to_bytes());
    if require_swapchain && !has_ext(ash::khr::swapchain::NAME) {
        gaps.push("VK_KHR_swapchain");
    }
    if !has_ext(ash::khr::dynamic_rendering::NAME) {
        gaps.push("VK_KHR_dynamic_rendering");
    }
    // synchronization2 is a baseline-tier requirement (ADR-027): the Vulkan
    // backend has no sync1 path. Devices lacking it fall back to wgpu.
    if !has_ext(ash::khr::synchronization2::NAME) {
        gaps.push("VK_KHR_synchronization2");
    }

    // Feature queries: the baseline tier needs dynamicRendering,
    // synchronization2, timelineSemaphore, and shaderDrawParameters
    // (SV_InstanceID compiles to SPIR-V DrawParameters).
    let mut vulkan11 = vk::PhysicalDeviceVulkan11Features::default();
    let mut vulkan12 = vk::PhysicalDeviceVulkan12Features::default();
    let mut dynamic_rendering = vk::PhysicalDeviceDynamicRenderingFeatures::default();
    let mut synchronization2 = vk::PhysicalDeviceSynchronization2Features::default();
    let mut features2 = vk::PhysicalDeviceFeatures2::default()
        .push_next(&mut vulkan11)
        .push_next(&mut vulkan12)
        .push_next(&mut dynamic_rendering)
        .push_next(&mut synchronization2);
    unsafe { instance.get_physical_device_features2(device, &mut features2) };

    if dynamic_rendering.dynamic_rendering == vk::FALSE {
        gaps.push("dynamicRendering feature");
    }
    if synchronization2.synchronization2 == vk::FALSE {
        gaps.push("synchronization2 feature");
    }
    if vulkan12.timeline_semaphore == vk::FALSE {
        gaps.push("timelineSemaphore feature");
    }
    if vulkan11.shader_draw_parameters == vk::FALSE {
        gaps.push("shaderDrawParameters feature");
    }

    gaps
}

/// Score a baseline-capable device under the given preference (ADR-028).
///
/// Pure so it is unit-testable without a Vulkan instance. Ordering
/// guarantees, in priority order:
/// - the preferred device type (discrete for high-performance, integrated
///   for low-power) always beats the other — the type gap (2000) exceeds any
///   possible memory (max 640) + resolution (max ~32) contribution;
/// - a software device (CPU/llvmpipe) never beats real hardware;
/// - among same-type devices, more device-local memory wins.
fn score_device(
    properties: &vk::PhysicalDeviceProperties,
    device_local_bytes: u64,
    preference: &crate::instance::AdapterPreference,
) -> u32 {
    use crate::instance::AdapterPreference;
    let low_power = matches!(preference, AdapterPreference::LowPower);
    let type_score = match properties.device_type {
        vk::PhysicalDeviceType::DISCRETE_GPU => {
            if low_power {
                1000
            } else {
                3000
            }
        }
        vk::PhysicalDeviceType::INTEGRATED_GPU => {
            if low_power {
                3000
            } else {
                1000
            }
        }
        vk::PhysicalDeviceType::VIRTUAL_GPU => 200,
        vk::PhysicalDeviceType::CPU => 0,
        _ => 100,
    };
    let memory_score = ((device_local_bytes >> 30).min(64) as u32) * 10;
    type_score + memory_score + properties.limits.max_image_dimension2_d / 1024
}

/// Size of the largest device-local memory heap.
fn device_local_heap_bytes(memory: &vk::PhysicalDeviceMemoryProperties) -> u64 {
    memory
        .memory_heaps
        .iter()
        .take(memory.memory_heap_count as usize)
        .filter(|h| h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
        .map(|h| h.size)
        .max()
        .unwrap_or(0)
}

/// Whether any queue family of the device can present on this platform.
///
/// Uses the surfaceless per-platform presentation-support queries, so
/// display-less compute accelerators (Tesla-class cards) are filtered out
/// *before* a window exists (ADR-028). Only Windows has a query that needs
/// no display connection; on other platforms every baseline device is
/// presumed presentable (compositors route across GPUs there anyway).
fn can_present(
    entry: &ash::Entry,
    instance: &ash::Instance,
    device: vk::PhysicalDevice,
    queue_family_count: u32,
) -> bool {
    #[cfg(target_os = "windows")]
    {
        let loader = ash::khr::win32_surface::Instance::new(entry, instance);
        (0..queue_family_count).any(|family| unsafe {
            loader.get_physical_device_win32_presentation_support(device, family)
        })
    }
    #[cfg(not(target_os = "windows"))]
    {
        let _ = (entry, instance, device, queue_family_count);
        true
    }
}

/// Select the best physical device that can run the baseline tier.
///
/// Filter (baseline tier, presentability, explicit-id match) then score
/// (ADR-028 preference policy) — the highest-scored survivor is always a
/// device that `vkCreateDevice` will accept.
///
/// `require_swapchain` mirrors the instance's surface support: a headless
/// instance cannot enable `VK_KHR_swapchain` (it depends on `VK_KHR_surface`),
/// so it must not be required of the device either.
pub fn select_physical_device(
    entry: &ash::Entry,
    instance: &ash::Instance,
    require_swapchain: bool,
    preference: &crate::instance::AdapterPreference,
    allow_maintenance9: bool,
) -> Result<SelectedDevice, GraphicsError> {
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

    let explicit_id = match preference {
        crate::instance::AdapterPreference::Explicit(id) => Some(id),
        _ => None,
    };

    let mut best: Option<(u32, SelectedDevice)> = None;
    // Baseline-capable devices that an explicit id did NOT match, named in
    // the error so a typo in REDLILIUM_ADAPTER is diagnosable immediately.
    let mut rejected_by_id: Vec<String> = Vec::new();

    for device in devices {
        let properties = unsafe { instance.get_physical_device_properties(device) };
        let device_name = unsafe { CStr::from_ptr(properties.device_name.as_ptr()) };

        let extensions = device_extension_set(instance, device)?;

        // Filter: below-baseline devices never reach scoring.
        let gaps = baseline_gaps(instance, device, require_swapchain, &extensions);
        if !gaps.is_empty() {
            log::info!(
                "Skipping GPU {:?}: missing {} (below baseline tier)",
                device_name,
                gaps.join(", ")
            );
            continue;
        }

        // Filter: a device that can never present is useless to a windowed
        // instance, regardless of its score.
        let queue_family_count =
            unsafe { instance.get_physical_device_queue_family_properties(device) }.len() as u32;
        if require_swapchain && !can_present(entry, instance, device, queue_family_count) {
            log::info!(
                "Skipping GPU {:?}: no presentation-capable queue family",
                device_name
            );
            continue;
        }

        // Filter: explicit adapter identity (ADR-028).
        if let Some(id) = explicit_id
            && !id.matches(
                properties.vendor_id,
                properties.device_id,
                &device_name.to_string_lossy(),
            )
        {
            rejected_by_id.push(format!(
                "{} ({:04x}:{:04x})",
                device_name.to_string_lossy(),
                properties.vendor_id,
                properties.device_id
            ));
            continue;
        }

        let memory = unsafe { instance.get_physical_device_memory_properties(device) };
        let score = score_device(&properties, device_local_heap_bytes(&memory), preference);

        log::info!(
            "Found GPU: {:?} (type: {:?}, score: {})",
            device_name,
            properties.device_type,
            score
        );

        if best.as_ref().is_none_or(|(s, _)| score > *s) {
            let features = unsafe { instance.get_physical_device_features(device) };
            let maintenance9 = allow_maintenance9
                && extensions.contains(super::maintenance9::EXTENSION_NAME.to_bytes())
                && super::maintenance9::feature_supported(instance, device);
            best = Some((
                score,
                SelectedDevice {
                    physical_device: device,
                    properties,
                    optional: OptionalFeatures {
                        sampler_anisotropy: features.sampler_anisotropy == vk::TRUE,
                        fill_mode_non_solid: features.fill_mode_non_solid == vk::TRUE,
                        maintenance9,
                    },
                    portability_subset: extensions.contains(b"VK_KHR_portability_subset".as_ref()),
                },
            ));
        }
    }

    best.map(|(_, selected)| selected).ok_or_else(|| {
        if let Some(id) = explicit_id {
            GraphicsError::InitializationFailed(format!(
                "Requested adapter {:?} not found among baseline-capable devices \
                 [{}] (check REDLILIUM_ADAPTER / adapter preference)",
                id,
                rejected_by_id.join(", ")
            ))
        } else {
            GraphicsError::InitializationFailed(
                "No GPU meets the baseline tier (Vulkan 1.2, dynamic rendering, \
                 synchronization2, timeline semaphores, shaderDrawParameters); \
                 see log for per-device gaps"
                    .to_string(),
            )
        }
    })
}

/// All extensions the device offers, as byte strings.
fn device_extension_set(
    instance: &ash::Instance,
    device: vk::PhysicalDevice,
) -> Result<HashSet<Vec<u8>>, GraphicsError> {
    let props = unsafe { instance.enumerate_device_extension_properties(device) }.map_err(|e| {
        GraphicsError::InitializationFailed(format!(
            "Failed to enumerate device extensions: {:?}",
            e
        ))
    })?;
    Ok(props
        .iter()
        .map(|p| {
            unsafe { CStr::from_ptr(p.extension_name.as_ptr()) }
                .to_bytes()
                .to_vec()
        })
        .collect())
}

/// The queues to create on the logical device.
///
/// Planned before device creation from the physical device's queue families:
/// the graphics queue (always), plus an async compute queue when the hardware
/// exposes one (#47 phase 4), plus a dedicated transfer queue when the
/// hardware exposes a transfer-only family (#89 — DMA engines: AMD SDMA,
/// NVIDIA copy engines). For async compute a dedicated compute-capable family
/// is preferred (real overlap on most discrete GPUs) with a second queue in
/// the graphics family as fallback; the transfer queue has no same-family
/// fallback — without DMA engines a "transfer queue" buys nothing, the
/// routing ladder falls back to async compute / graphics instead. Missing
/// queues are a first-class, fully supported mode (MoltenVK's default
/// configuration, for one, exposes a single queue).
#[derive(Debug, Clone, Copy)]
pub struct QueuePlan {
    /// Queue family of the graphics queue (queue index 0 within it).
    pub graphics_family: u32,
    /// `(family, queue index)` of the async compute queue, if available.
    pub async_compute: Option<(u32, u32)>,
    /// `(family, queue index)` of the dedicated transfer queue, if available.
    pub transfer: Option<(u32, u32)>,
}

/// Plan which queues to create on the device.
pub fn plan_queues(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> Result<QueuePlan, GraphicsError> {
    let queue_families =
        unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
    let plan = plan_queues_from_families(&queue_families)?;

    match plan.async_compute {
        Some((family, index)) => log::info!(
            "Async compute queue planned: family {family}, queue {index}{}",
            if family == plan.graphics_family {
                " (graphics family)"
            } else {
                " (dedicated family)"
            }
        ),
        None => log::info!("No async compute queue available; single-queue mode"),
    }
    match plan.transfer {
        Some((family, index)) => {
            log::info!("Dedicated transfer queue planned: family {family}, queue {index}")
        }
        None => log::info!("No dedicated transfer queue; transfer graphs fall back"),
    }

    Ok(plan)
}

/// Pure queue planning from the queried family properties (unit-testable).
fn plan_queues_from_families(
    queue_families: &[vk::QueueFamilyProperties],
) -> Result<QueuePlan, GraphicsError> {
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

    // A transfer-only family = the DMA engines. Exclude specialized families
    // (video decode/encode, optical flow) that also advertise TRANSFER, and
    // require 1×1×1 image-transfer granularity so buffer→image copies of
    // arbitrary regions are legal — a coarser-granularity family would need
    // per-copy alignment validation, which no target device has warranted.
    let specialized = vk::QueueFlags::GRAPHICS
        | vk::QueueFlags::COMPUTE
        | vk::QueueFlags::VIDEO_DECODE_KHR
        | vk::QueueFlags::VIDEO_ENCODE_KHR
        | vk::QueueFlags::OPTICAL_FLOW_NV;
    let transfer = queue_families
        .iter()
        .enumerate()
        .find(|(_, family)| {
            family.queue_flags.contains(vk::QueueFlags::TRANSFER)
                && !family.queue_flags.intersects(specialized)
                && family.queue_count > 0
                && family.min_image_transfer_granularity
                    == vk::Extent3D {
                        width: 1,
                        height: 1,
                        depth: 1,
                    }
        })
        .map(|(family, _)| (family as u32, 0));

    Ok(QueuePlan {
        graphics_family,
        async_compute,
        transfer,
    })
}

/// Create a logical device from a selected physical device.
///
/// Baseline-tier requirements are enabled unconditionally — selection already
/// verified them. Optional features are enabled only when the device reported
/// them (VK-M10): a GPU without `fillModeNonSolid` loses wireframe pipelines
/// instead of failing `vkCreateDevice` with `ERROR_FEATURE_NOT_PRESENT`.
pub fn create_logical_device(
    instance: &ash::Instance,
    selected: &SelectedDevice,
    plan: &QueuePlan,
    enable_swapchain: bool,
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
    // The transfer family is transfer-only by construction (#89), so it can
    // never coincide with the graphics family or a compute-capable family.
    if let Some((family, _)) = plan.transfer {
        queue_create_infos.push(
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(family)
                .queue_priorities(&queue_priorities[..1]),
        );
    }

    // Baseline-tier extensions (verified present during selection).
    let mut device_extensions = vec![
        ash::khr::dynamic_rendering::NAME.as_ptr(),
        ash::khr::synchronization2::NAME.as_ptr(),
    ];
    if enable_swapchain {
        device_extensions.push(ash::khr::swapchain::NAME.as_ptr());
    }
    // The spec requires enabling portability_subset whenever the device
    // lists it (MoltenVK).
    if selected.portability_subset {
        device_extensions.push(c"VK_KHR_portability_subset".as_ptr());
    }
    // maintenance9 (#88): optional queue-family ownership transfers. Enabled
    // whenever supported; the image fast path additionally checks the
    // per-queue-family transfer property (see `VulkanBackend::with_params`).
    if selected.optional.maintenance9 {
        device_extensions.push(super::maintenance9::EXTENSION_NAME.as_ptr());
    }

    // Optional features, enabled only where supported.
    let features = vk::PhysicalDeviceFeatures::default()
        .sampler_anisotropy(selected.optional.sampler_anisotropy)
        .fill_mode_non_solid(selected.optional.fill_mode_non_solid);

    // Enable dynamic rendering via extension features (works on Vulkan 1.2 with extension)
    // This is compatible with MoltenVK which only supports Vulkan 1.2
    let mut dynamic_rendering_features =
        vk::PhysicalDeviceDynamicRenderingFeatures::default().dynamic_rendering(true);

    // synchronization2: the backend records all barriers via
    // vkCmdPipelineBarrier2 and submits via vkQueueSubmit2 (ADR-027, #94).
    let mut synchronization2_features =
        vk::PhysicalDeviceSynchronization2Features::default().synchronization2(true);

    // shaderDrawParameters: shaders that read SV_InstanceID compile to SPIR-V
    // using the DrawParameters capability, which requires this feature.
    let mut vulkan11_features =
        vk::PhysicalDeviceVulkan11Features::default().shader_draw_parameters(true);

    // timelineSemaphore: the queue timeline that backs frame fences and (with
    // multi-queue, #47) cross-queue waits.
    let mut vulkan12_features =
        vk::PhysicalDeviceVulkan12Features::default().timeline_semaphore(true);

    // maintenance9 feature struct — chained only when the device supports it.
    let mut maintenance9_features =
        super::maintenance9::PhysicalDeviceMaintenance9FeaturesKHR::enabled();

    let mut create_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_create_infos)
        .enabled_extension_names(&device_extensions)
        .enabled_features(&features)
        .push_next(&mut dynamic_rendering_features)
        .push_next(&mut synchronization2_features)
        .push_next(&mut vulkan11_features)
        .push_next(&mut vulkan12_features);
    if selected.optional.maintenance9 {
        create_info = create_info.push_next(&mut maintenance9_features);
    }

    let device = unsafe { instance.create_device(selected.physical_device, &create_info, None) }
        .map_err(|e| {
            GraphicsError::InitializationFailed(format!("Failed to create logical device: {:?}", e))
        })?;

    Ok(device)
}

/// Assemble the engine-facing capabilities from everything queried about the
/// device (ADR-027). The single source of truth downstream clamps against —
/// nothing here is fabricated.
pub fn device_capabilities(
    instance: &ash::Instance,
    selected: &SelectedDevice,
    plan: &QueuePlan,
) -> crate::device::DeviceCapabilities {
    let limits = &selected.properties.limits;

    // Largest device-local heap: the only honest creation-size bound Vulkan
    // gives for plain buffers (there is no core "max buffer size" limit).
    let memory =
        unsafe { instance.get_physical_device_memory_properties(selected.physical_device) };
    let max_buffer_size = match device_local_heap_bytes(&memory) {
        0 => 1 << 30,
        bytes => bytes,
    };

    // Per-pass GPU timestamps (#95) require the graphics queue family to expose
    // a non-zero `timestampValidBits` and a non-zero tick period. `timestamp_
    // period` is nanoseconds-per-tick; a value of 0 means timestamps are
    // unsupported. This is queried, never fabricated (ADR-027).
    let queue_families =
        unsafe { instance.get_physical_device_queue_family_properties(selected.physical_device) };
    let gpu_timestamps = limits.timestamp_period > 0.0
        && queue_families
            .get(plan.graphics_family as usize)
            .is_some_and(|f| f.timestamp_valid_bits > 0);

    crate::device::DeviceCapabilities {
        // Selection passing the baseline filter IS the tier detection today;
        // higher rungs (bindless, ray tracing) will extend this when their
        // render paths exist (ADR-027).
        tier: crate::device::DeviceTier::Baseline,
        max_texture_dimension: limits.max_image_dimension2_d,
        max_buffer_size,
        max_sampler_anisotropy: if selected.optional.sampler_anisotropy {
            limits.max_sampler_anisotropy as u16
        } else {
            1
        },
        // VkSampleCountFlagBits values are the counts themselves, matching
        // the mask encoding of `DeviceCapabilities::sample_count_mask`.
        sample_count_mask: (limits.framebuffer_color_sample_counts
            & limits.framebuffer_depth_sample_counts)
            .as_raw(),
        wireframe: selected.optional.fill_mode_non_solid,
        async_compute: plan.async_compute.is_some(),
        transfer_queue: plan.transfer.is_some(),
        compute_shaders: true,
        gpu_timestamps,
    }
}

/// Engine-facing adapter info for the selected device.
pub fn adapter_info(selected: &SelectedDevice) -> crate::instance::AdapterInfo {
    let name = unsafe { CStr::from_ptr(selected.properties.device_name.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    let device_type = match selected.properties.device_type {
        vk::PhysicalDeviceType::DISCRETE_GPU => crate::instance::AdapterType::Discrete,
        vk::PhysicalDeviceType::INTEGRATED_GPU => crate::instance::AdapterType::Integrated,
        vk::PhysicalDeviceType::CPU => crate::instance::AdapterType::Software,
        _ => crate::instance::AdapterType::Unknown,
    };
    crate::instance::AdapterInfo {
        name,
        vendor: crate::instance::vendor_name(selected.properties.vendor_id),
        device_type,
        vendor_id: selected.properties.vendor_id,
        device_id: selected.properties.device_id,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instance::AdapterPreference;

    const GIB: u64 = 1 << 30;

    fn props(device_type: vk::PhysicalDeviceType, max_dim: u32) -> vk::PhysicalDeviceProperties {
        vk::PhysicalDeviceProperties {
            device_type,
            limits: vk::PhysicalDeviceLimits {
                max_image_dimension2_d: max_dim,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// ADR-028: under the default policy the discrete GPU always beats the
    /// integrated one, no matter how much memory the iGPU claims (UMA iGPUs
    /// report the whole system RAM as device-local).
    #[test]
    fn high_performance_discrete_beats_integrated() {
        let discrete = score_device(
            &props(vk::PhysicalDeviceType::DISCRETE_GPU, 16384),
            8 * GIB,
            &AdapterPreference::Auto,
        );
        let integrated = score_device(
            &props(vk::PhysicalDeviceType::INTEGRATED_GPU, 16384),
            128 * GIB,
            &AdapterPreference::Auto,
        );
        assert!(discrete > integrated);
    }

    /// ADR-028: LowPower inverts the type ranking.
    #[test]
    fn low_power_integrated_beats_discrete() {
        let discrete = score_device(
            &props(vk::PhysicalDeviceType::DISCRETE_GPU, 16384),
            128 * GIB,
            &AdapterPreference::LowPower,
        );
        let integrated = score_device(
            &props(vk::PhysicalDeviceType::INTEGRATED_GPU, 16384),
            8 * GIB,
            &AdapterPreference::LowPower,
        );
        assert!(integrated > discrete);
    }

    /// A software renderer (llvmpipe) must never beat real hardware, even
    /// with a huge host-memory "device-local" heap and max limits.
    #[test]
    fn software_never_beats_hardware() {
        let llvmpipe = score_device(
            &props(vk::PhysicalDeviceType::CPU, 16384),
            256 * GIB,
            &AdapterPreference::Auto,
        );
        let integrated = score_device(
            &props(vk::PhysicalDeviceType::INTEGRATED_GPU, 4096),
            GIB,
            &AdapterPreference::Auto,
        );
        assert!(integrated > llvmpipe);
    }

    fn family(flags: vk::QueueFlags, count: u32, granularity: u32) -> vk::QueueFamilyProperties {
        vk::QueueFamilyProperties {
            queue_flags: flags,
            queue_count: count,
            min_image_transfer_granularity: vk::Extent3D {
                width: granularity,
                height: granularity,
                depth: granularity,
            },
            ..Default::default()
        }
    }

    /// #89: a typical discrete-GPU family layout (graphics+compute, dedicated
    /// compute, transfer-only SDMA) plans all three queues.
    #[test]
    fn plan_finds_dedicated_transfer_family() {
        let families = [
            family(
                vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER,
                1,
                1,
            ),
            family(vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER, 4, 1),
            family(
                vk::QueueFlags::TRANSFER | vk::QueueFlags::SPARSE_BINDING,
                2,
                1,
            ),
        ];
        let plan = plan_queues_from_families(&families).unwrap();
        assert_eq!(plan.graphics_family, 0);
        assert_eq!(plan.async_compute, Some((1, 0)));
        assert_eq!(plan.transfer, Some((2, 0)));
    }

    /// #89: video decode/encode families advertise TRANSFER but are not DMA
    /// engines for general streaming — they must not be picked.
    #[test]
    fn plan_skips_video_families() {
        let families = [
            family(
                vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER,
                2,
                1,
            ),
            family(
                vk::QueueFlags::VIDEO_DECODE_KHR | vk::QueueFlags::TRANSFER,
                1,
                1,
            ),
        ];
        let plan = plan_queues_from_families(&families).unwrap();
        assert_eq!(plan.transfer, None);
        // Async compute still falls back to the second graphics-family queue.
        assert_eq!(plan.async_compute, Some((0, 1)));
    }

    /// #89: a transfer family with coarse `minImageTransferGranularity` would
    /// restrict image copies; it is skipped rather than validated per copy.
    #[test]
    fn plan_skips_coarse_granularity_transfer_family() {
        let families = [
            family(
                vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER,
                1,
                1,
            ),
            family(vk::QueueFlags::TRANSFER, 1, 8),
        ];
        let plan = plan_queues_from_families(&families).unwrap();
        assert_eq!(plan.transfer, None);
    }

    /// Single-family devices (MoltenVK default) plan no secondary queues —
    /// the first-class fallback mode.
    #[test]
    fn plan_single_family_no_secondary_queues() {
        let families = [family(
            vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER,
            1,
            1,
        )];
        let plan = plan_queues_from_families(&families).unwrap();
        assert_eq!(plan.async_compute, None);
        assert_eq!(plan.transfer, None);
    }

    /// Same type: more device-local memory wins.
    #[test]
    fn memory_breaks_ties_within_type() {
        let small = score_device(
            &props(vk::PhysicalDeviceType::DISCRETE_GPU, 16384),
            8 * GIB,
            &AdapterPreference::HighPerformance,
        );
        let large = score_device(
            &props(vk::PhysicalDeviceType::DISCRETE_GPU, 16384),
            24 * GIB,
            &AdapterPreference::HighPerformance,
        );
        assert!(large > small);
    }
}
