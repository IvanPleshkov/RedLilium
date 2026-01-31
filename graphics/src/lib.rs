//! # RedLilium Graphics
//!
//! Custom rendering engine for RedLilium built around an abstract render graph.
//!
//! ## Overview
//!
//! This crate provides:
//! - [`RenderGraph`] - Declarative description of render passes and dependencies
//! - [`Backend`] - Trait for graphics backend implementations
//! - [`scene`] - ECS to render graph integration
//! - Multiple backend support: Vulkan, wgpu, and Dummy (for testing)
//!
//! ## Example
//!
//! ```ignore
//! use redlilium_graphics::{RenderGraph, Backend};
//!
//! let mut renderer = SceneRenderer::new();
//! renderer.begin_frame();
//! // Extract from ECS, prepare, render...
//! renderer.end_frame();
//! ```

pub mod backend;
pub mod graph;
pub mod scene;
pub mod types;

// Re-export main types for convenience
pub use backend::{Backend, BackendError, DummyBackend};
pub use graph::{PassHandle, RenderGraph, RenderPass, ResourceHandle};
pub use scene::{
    CameraRenderContext, CameraSystem, ExtractedCamera, ExtractedMaterial, ExtractedMesh,
    ExtractedTransform, RenderWorld,
};
pub use types::{
    BufferDescriptor, BufferUsage, ClearValue, Extent3d, SamplerDescriptor, TextureDescriptor,
    TextureFormat, TextureUsage,
};

/// Graphics library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Initialize the graphics subsystem.
///
/// This should be called before using any graphics functionality.
pub fn init() {
    log::info!("RedLilium Graphics v{} initialized", VERSION);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_render_graph_creation() {
        let graph = RenderGraph::new();
        assert!(graph.passes().is_empty());
    }

    #[test]
    fn test_dummy_backend() {
        let backend = DummyBackend::new();
        assert!(backend.name() == "Dummy");
    }
}
