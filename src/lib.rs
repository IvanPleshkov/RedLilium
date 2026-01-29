//! Graphics Engine - A render graph-based graphics engine with Forward+ lighting
//!
//! This engine supports two backends:
//! - **wgpu**: Cross-platform, high-level GPU abstraction (including web via WebGPU)
//! - **Vulkan**: Direct Vulkan API via ash for maximum control (native only)
//!
//! # Features
//! - Render graph system for declarative render pass management
//! - Forward+ (tiled forward) rendering pipeline
//! - Post-processing effects (bloom, tonemapping)
//! - Flexible resource management
//! - Web support via WebAssembly and WebGPU
//! - Entity Component System (ECS) based scene management using Bevy ECS

pub mod backend;
pub mod egui_integration;
pub mod engine;
pub mod pipeline;
pub mod render_graph;
pub mod resources;
pub mod scene;
pub mod window;

// Re-export Bevy ECS prelude for users
pub use bevy_ecs::prelude::*;

// Web-specific modules
#[cfg(target_arch = "wasm32")]
pub mod web;

#[cfg(target_arch = "wasm32")]
mod web_demo;

pub use egui_integration::WgpuEguiIntegration;
#[cfg(not(target_arch = "wasm32"))]
pub use egui_integration::VulkanEguiIntegration;
pub use engine::Engine;
pub use window::Window;

// Re-export wgpu backend for direct access
pub use backend::wgpu_backend::WgpuBackend;

/// Backend selection for the graphics engine
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BackendType {
    /// wgpu backend - cross-platform, easier to use (supports web)
    #[default]
    Wgpu,
    /// Vulkan backend via ash - maximum control (native only)
    Vulkan,
}

/// Configuration for initializing the graphics engine
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Window title
    pub title: String,
    /// Initial window width
    pub width: u32,
    /// Initial window height
    pub height: u32,
    /// Which backend to use
    pub backend: BackendType,
    /// Enable vsync
    pub vsync: bool,
    /// Tile size for Forward+ light culling (in pixels)
    pub tile_size: u32,
    /// Maximum number of lights supported
    pub max_lights: u32,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            title: "Graphics Engine".to_string(),
            width: 1280,
            height: 720,
            backend: BackendType::Wgpu,
            vsync: true,
            tile_size: 16,
            max_lights: 1024,
        }
    }
}

// Web initialization helper
#[cfg(target_arch = "wasm32")]
pub fn init_web_logging() {
    // Set up panic hook for better error messages in console
    console_error_panic_hook::set_once();
    // Set up console logging for web
    console_log::init_with_level(log::Level::Info).expect("Failed to initialize logger");
}
