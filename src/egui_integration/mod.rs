//! egui GUI integration
//!
//! Provides egui rendering support for both wgpu and Vulkan backends.

mod wgpu;

#[cfg(not(target_arch = "wasm32"))]
mod vulkan;

pub use self::wgpu::WgpuEguiIntegration;

#[cfg(not(target_arch = "wasm32"))]
pub use self::vulkan::VulkanEguiIntegration;
