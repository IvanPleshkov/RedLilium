// The wgpu type graph is deep; auto-trait (Send/Sync) evaluation for our
// resource types (e.g. the `assert_impl_all!(BindingGroup: ...)` checks)
// exceeds the default recursion limit of 128.
#![recursion_limit = "256"]
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
//! | Schedule | [`FrameSchedule`] | Submits the frame's render graphs |
//! | Pipeline | [`FramePipeline`] | Multiple frames in flight for CPU-GPU overlap |
//!
//! **Creation hierarchy:**
//! - [`GraphicsDevice::create_pipeline`] → [`FramePipeline`]
//! - [`FramePipeline::begin_frame`] → [`FrameSchedule`]
//!
//! **Synchronization primitives:**
//! - Pass → Pass: Barriers (automatic, within the graph)
//! - Graph → Graph: Submission order on the single graphics queue (barriers
//!   cross submits automatically; see `FrameSchedule::submit`)
//! - Frame → Frame: [`Fence`] (CPU-GPU sync across frames, one per submit)
//!
//! For detailed architecture documentation, see `docs/ARCHITECTURE.md`.
//!
//! ## Example
//!
//! ```ignore
//! use redlilium_graphics::{GraphicsInstance, GraphicsPass};
//!
//! let instance = GraphicsInstance::new()?;
//! let device = instance.create_device()?;
//! let mut pipeline = device.create_pipeline(2);  // 2 frames in flight
//!
//! // Frame loop
//! while running {
//!     let mut schedule = pipeline.begin_frame();
//!
//!     // Acquire a pooled graph (avoids per-frame allocation)
//!     let mut graph = schedule.acquire_graph();
//!     let geometry = graph.add_graphics_pass(GraphicsPass::new("geometry".into()));
//!     let lighting = graph.add_graphics_pass(GraphicsPass::new("lighting".into()));
//!     graph.add_dependency(lighting, geometry);
//!
//!     schedule.submit(graph); // graphs execute in submission order
//!     pipeline.end_frame(schedule);
//! }
//!
//! pipeline.wait_idle();  // Graceful shutdown
//! ```

mod as_compaction;
pub mod backend;
pub mod bindless;
pub mod compiler;
pub mod device;
pub mod egui;
pub mod error;
pub mod graph;
pub mod instance;
pub mod materials;
pub mod mesh;
pub mod pipeline;
pub mod resize;
pub mod resources;
pub mod scheduler;
pub mod shader;
pub mod swapchain;
pub mod types;

// Re-export main types for convenience
pub use bindless::{BINDLESS_SAMPLERS_BINDING, BINDLESS_TEXTURES_BINDING, BindlessSlots};
pub use device::{
    DeviceCapabilities, DeviceTier, FrameGpuTimings, GpuMemoryStats, GraphicsDevice, HeapStats,
    ResourceCounts, SubmitTiming,
};
pub use error::GraphicsError;
pub use graph::{
    AccelerationStructureBuild, AccelerationStructureBuildPass, BufferCopyRegion,
    BufferTextureCopyRegion, BufferTextureLayout, ColorAttachment, CompiledGraph, ComputePass,
    DepthStencilAttachment, DrawCommand, GraphError, GraphicsPass, IndirectDrawCommand, LoadOp,
    MeshTasksDrawCommand, MeshTasksIndirectDrawCommand, Pass, PassHandle, QueuePreference,
    RenderGraph, RenderGraphCompilationMode, RenderTarget, RenderTargetConfig, StoreOp,
    TextureCopyLocation, TextureCopyRegion, TextureOrigin, TransferConfig, TransferOperation,
    TransferPass,
    resource_usage::{PassResourceUsage, TextureAccessMode, TextureUsageDecl},
};
pub use instance::{
    AdapterId, AdapterInfo, AdapterPreference, AdapterType, BackendType, BreadcrumbsMode,
    GraphicsInstance, InstanceParameters, WgpuBackendType,
};
pub use materials::{
    BindingGroup, BindingGroupDescriptor, BindingLayout, BindingLayoutEntry, BindingType,
    BlendComponent, BlendFactor, BlendOperation, BlendState, BoundResource, CullMode,
    DepthConvention, DepthState, FrontFace, Material, MaterialDescriptor, MaterialInstance,
    PolygonMode, RasterState, SampledDepthLayout, ShaderSource, ShaderSourceLanguage, ShaderStage,
    ShaderStageFlags, UpdateRate,
};
pub use mesh::{
    CpuMesh, IndexFormat, Mesh, MeshDescriptor, PrimitiveTopology, VertexAttribute,
    VertexAttributeFormat, VertexAttributeSemantic, VertexBufferLayout, VertexLayout,
    VertexStepMode,
};
pub use pipeline::{FramePipeline, MAX_FRAMES_IN_FLIGHT};
pub use resize::{ResizeEvent, ResizeManager, ResizeStrategy};
pub use resources::{
    AccelBuildSizes, Blas, BlasDescriptor, BlasTriangles, Buffer, RingAllocation, RingBuffer,
    Sampler, Texture, Tlas, TlasDescriptor, TlasInstance,
};
pub use scheduler::{Fence, FenceStatus, FrameSchedule, SubmitHandle};
pub use shader::{
    ShaderLibrary, ShaderVariantSpace, VariantError, VariantKey, VariantSelector, VariantValue,
};
pub use swapchain::{PresentMode, Surface, SurfaceConfiguration, SurfaceTexture};
pub use types::{
    AddressMode, BufferDescriptor, BufferUsage, ClearValue, CompareFunction, CompressionFamily,
    CpuSampler, CpuTexture, DrawIndexedIndirectArgs, DrawIndirectArgs, DrawMeshTasksIndirectArgs,
    Extent3d, FilterMode, SamplerDescriptor, ScissorRect, TextureDescriptor, TextureDimension,
    TextureFormat, TextureUsage, Viewport,
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
