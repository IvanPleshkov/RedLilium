//! # RedLilium Graphics
//!
//! Custom rendering engine for RedLilium built around an abstract render graph.
//!
//! ## Overview
//!
//! This crate provides:
//! - [`GraphicsInstance`] - Top-level graphics system entry point
//! - [`GraphicsDevice`] - Device for creating GPU resources
//! - [`RenderGraph`] - Declarative description of render passes and dependencies
//! - [`Surface`] - Swapchain abstraction for presenting to windows
//! - [`scene`] - ECS to render graph integration
//!
//! ## Example
//!
//! ```ignore
//! use redlilium_graphics::{GraphicsInstance, RenderGraph};
//!
//! let instance = GraphicsInstance::new()?;
//! let device = instance.create_device()?;
//!
//! let mut graph = RenderGraph::new();
//! let geometry = graph.add_graphics_pass("geometry");
//! let lighting = graph.add_graphics_pass("lighting");
//! graph.add_dependency(lighting, geometry);
//! ```

pub mod device;
pub mod error;
pub mod graph;
pub mod instance;
pub mod materials;
pub mod resources;
pub mod scene;
pub mod swapchain;
pub mod types;

// Re-export main types for convenience
pub use device::{DeviceCapabilities, GraphicsDevice};
pub use error::GraphicsError;
pub use graph::{
    BufferCopyRegion, BufferTextureCopyRegion, BufferTextureLayout, ColorAttachment, ComputePass,
    DepthStencilAttachment, GraphicsPass, LoadOp, Pass, PassHandle, RenderGraph, RenderTarget,
    RenderTargetConfig, StoreOp, TextureCopyLocation, TextureCopyRegion, TextureOrigin,
    TransferConfig, TransferOperation, TransferPass,
};
pub use instance::{AdapterInfo, AdapterType, GraphicsInstance};
pub use materials::{
    BindingGroup, BindingLayout, BindingLayoutEntry, BindingType, BoundResource, Material,
    MaterialDescriptor, MaterialInstance, ShaderSource, ShaderStage, ShaderStageFlags,
};
pub use resources::{Buffer, Sampler, Texture};
pub use scene::{
    CameraRenderContext, CameraSystem, ExtractedCamera, ExtractedMaterial, ExtractedMesh,
    ExtractedTransform, RenderWorld,
};
pub use swapchain::{PresentMode, Surface, SurfaceConfiguration, SurfaceTexture};
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
    fn test_instance_creation() {
        let instance = GraphicsInstance::new().unwrap();
        assert_eq!(instance.device_count(), 0);
    }

    #[test]
    fn test_device_creation() {
        let instance = GraphicsInstance::new().unwrap();
        let device = instance.create_device().unwrap();
        assert!(!device.name().is_empty());
    }

    #[test]
    fn test_resource_creation() {
        let instance = GraphicsInstance::new().unwrap();
        let device = instance.create_device().unwrap();

        let buffer = device
            .create_buffer(&BufferDescriptor::new(1024, BufferUsage::VERTEX))
            .unwrap();
        assert_eq!(buffer.size(), 1024);

        let texture = device
            .create_texture(&TextureDescriptor::new_2d(
                512,
                512,
                TextureFormat::Rgba8Unorm,
                TextureUsage::TEXTURE_BINDING,
            ))
            .unwrap();
        assert_eq!(texture.width(), 512);
    }
}
