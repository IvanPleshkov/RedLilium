//! GPU-memory statistics for the Vulkan backend (#98).
//!
//! Two data sources, both cheap to poll once per frame:
//!
//! 1. **Driver heap budgets** — `VK_EXT_memory_budget` chained into
//!    `vkGetPhysicalDeviceMemoryProperties2`. Per heap: `heapBudget`/`heapUsage`
//!    plus the core `VkMemoryHeap::size` and `DEVICE_LOCAL` flag. Usage/budget
//!    are process-wide (system-wide for budget) — the point is to show headroom,
//!    not just our own footprint. The spec permits these to change at any
//!    moment, so they are sampled every frame in `advance_frame` and never
//!    cached across frames.
//! 2. **Allocator totals** — `gpu_allocator::vulkan::Allocator::generate_report`
//!    gives bytes allocated to live resources, bytes reserved in memory blocks,
//!    and the live-allocation count.
//!
//! The engine's own live-resource counts ([`GpuMemoryStats::resources`]) are
//! filled a layer up in [`GraphicsDevice::latest_memory_stats`](crate::device::GraphicsDevice::latest_memory_stats).

use ash::vk;
use gpu_allocator::vulkan::Allocator;

use crate::device::{GpuMemoryStats, HeapStats};

/// Sample driver heap budgets and the allocator's totals into a
/// [`GpuMemoryStats`]. Called on the render thread from `advance_frame`.
/// `resources` is left at its default here (filled at the device layer).
pub fn sample(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    memory_budget: bool,
    allocator: &Allocator,
) -> GpuMemoryStats {
    let heaps = query_heaps(instance, physical_device, memory_budget);
    let report = allocator.generate_report();
    GpuMemoryStats {
        heaps,
        allocator_allocated: report.total_allocated_bytes,
        allocator_reserved: report.total_capacity_bytes,
        allocation_count: report.allocations.len(),
        resources: crate::device::ResourceCounts::default(),
    }
}

/// Per-heap sizes + `DEVICE_LOCAL` flags from core memory properties, plus
/// budget/usage from the EXT struct when `memory_budget` is enabled.
fn query_heaps(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    memory_budget: bool,
) -> Vec<HeapStats> {
    let mut budget_props = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
    let props2 = vk::PhysicalDeviceMemoryProperties2::default();
    // Chain the budget struct only when the extension is enabled — chaining it
    // otherwise is invalid usage.
    let mut props2 = if memory_budget {
        props2.push_next(&mut budget_props)
    } else {
        props2
    };
    unsafe { instance.get_physical_device_memory_properties2(physical_device, &mut props2) };

    // Copy the core properties out (they are `Copy`) so `props2`'s mutable
    // borrow of `budget_props` ends and the budget arrays can be read below.
    let mem = props2.memory_properties;

    (0..mem.memory_heap_count as usize)
        .map(|i| {
            let heap = mem.memory_heaps[i];
            let (budget, usage) = if memory_budget {
                (
                    Some(budget_props.heap_budget[i]),
                    Some(budget_props.heap_usage[i]),
                )
            } else {
                (None, None)
            };
            HeapStats {
                index: i as u32,
                device_local: heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL),
                size: heap.size,
                budget,
                usage,
            }
        })
        .collect()
}
