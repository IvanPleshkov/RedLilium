//! Backend abstraction layer
//!
//! Provides common traits and types that both wgpu and Vulkan backends implement.

pub mod traits;
pub mod types;
pub mod wgpu_backend;

// Vulkan backend is only available on native platforms
#[cfg(not(target_arch = "wasm32"))]
pub mod vulkan;

pub use traits::*;
pub use types::*;
