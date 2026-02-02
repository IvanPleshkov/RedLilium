//! # RedLilium Graphics
//!
//! Custom rendering engine for RedLilium built around an abstract render graph.
//!
//! ## Overview
//!
//! This crate provides:
//! - [`GraphicsInstance`] - Top-level graphics system entry point
//! - [`GraphicsDevice`] - Device for creating GPU resources and pipelines
//! - [`FramePipeline`] - Frame-level CPU-GPU synchronization
//! - [`FrameSchedule`] - Streaming graph submission within a frame
//! - [`RenderGraph`] - Declarative description of render passes and dependencies
//! - [`Surface`] - Swapchain abstraction for presenting to windows
//!
//! ## Rendering Architecture
//!
//! The rendering system is organized in layers, from low-level to high-level:
//!
//! | Layer | Type | Purpose |
//! |-------|------|---------|
//! | Pass | [`GraphicsPass`], [`TransferPass`], [`ComputePass`] | Single unit of GPU work |
//! | Graph | [`RenderGraph`] + [`PassHandle`] | Passes and dependencies for one task |
//! | Schedule | [`FrameSchedule`] + [`GraphHandle`] | Streaming submission of graphs in one frame |
//! | Pipeline | [`FramePipeline`] | Multiple frames in flight for CPU-GPU overlap |
//!
//! **Creation hierarchy:**
//! - [`GraphicsDevice::create_pipeline`] → [`FramePipeline`]
//! - [`FramePipeline::begin_frame`] → [`FrameSchedule`]
//!
//! **Synchronization primitives:**
//! - Pass → Pass: Barriers (automatic, within a graph)
//! - Graph → Graph: [`Semaphore`] (GPU-GPU sync within a frame)
//! - Frame → Frame: [`Fence`] (CPU-GPU sync across frames)
//!
//! For detailed architecture documentation, see `docs/ARCHITECTURE.md`.
//!
//! ## Example
//!
//! ```ignore
//! use redlilium_graphics::{GraphicsInstance, GraphicsPass, RenderGraph};
//!
//! let instance = GraphicsInstance::new()?;
//! let device = instance.create_device()?;
//! let mut pipeline = device.create_pipeline(2);  // 2 frames in flight
//!
//! // Build a render graph
//! let mut graph = RenderGraph::new();
//! let geometry = graph.add_graphics_pass(GraphicsPass::new("geometry".into()));
//! let lighting = graph.add_graphics_pass(GraphicsPass::new("lighting".into()));
//! graph.add_dependency(lighting, geometry);
//!
//! // Frame loop
//! while running {
//!     let mut schedule = pipeline.begin_frame();
//!     let main = schedule.submit("main", graph.compile()?, &[]);
//!     schedule.present("present", post_graph.compile()?, &[main]);
//!     pipeline.end_frame(schedule);
//! }
//!
//! pipeline.wait_idle();  // Graceful shutdown
//! ```

pub mod backend;
pub mod compiler;
pub mod device;
pub mod error;
pub mod graph;
pub mod instance;
pub mod materials;
pub mod mesh;
pub mod pipeline;
pub mod resize;
pub mod resources;
pub mod scheduler;
pub mod swapchain;
pub mod types;

// Re-export main types for convenience
pub use device::{DeviceCapabilities, GraphicsDevice};
pub use error::GraphicsError;
pub use graph::{
    BufferCopyRegion, BufferTextureCopyRegion, BufferTextureLayout, ColorAttachment, CompiledGraph,
    ComputePass, DepthStencilAttachment, DrawCommand, GraphError, GraphicsPass,
    IndirectDrawCommand, LoadOp, Pass, PassHandle, RenderGraph, RenderTarget, RenderTargetConfig,
    StoreOp, TextureCopyLocation, TextureCopyRegion, TextureOrigin, TransferConfig,
    TransferOperation, TransferPass,
    resource_usage::{PassResourceUsage, TextureAccessMode, TextureUsageDecl},
};
pub use instance::{
    AdapterInfo, AdapterType, BackendType, GraphicsInstance, InstanceParameters, WgpuBackendType,
};
pub use materials::{
    BindingGroup, BindingLayout, BindingLayoutEntry, BindingType, BoundResource, Material,
    MaterialDescriptor, MaterialInstance, ShaderSource, ShaderStage, ShaderStageFlags,
};
pub use mesh::{
    IndexFormat, Mesh, MeshDescriptor, PrimitiveTopology, VertexAttribute, VertexAttributeFormat,
    VertexAttributeSemantic, VertexBufferLayout, VertexLayout, VertexStepMode,
};
pub use pipeline::FramePipeline;
pub use resize::{ResizeEvent, ResizeManager, ResizeStrategy};
pub use resources::{Buffer, Sampler, Texture};
pub use scheduler::{Fence, FenceStatus, FrameSchedule, GraphHandle, Semaphore};
pub use swapchain::{PresentMode, Surface, SurfaceConfiguration, SurfaceTexture};
pub use types::{
    AddressMode, BufferDescriptor, BufferUsage, ClearValue, CompareFunction,
    DrawIndexedIndirectArgs, DrawIndirectArgs, Extent3d, FilterMode, SamplerDescriptor,
    ScissorRect, TextureDescriptor, TextureDimension, TextureFormat, TextureUsage, Viewport,
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
