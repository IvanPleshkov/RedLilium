//! Core backend abstraction traits
//!
//! These traits define the interface that both wgpu and Vulkan backends must implement.

use crate::backend::types::*;
use std::sync::Arc;
use thiserror::Error;

/// Backend error type
#[derive(Error, Debug)]
pub enum BackendError {
    #[error("Failed to initialize backend: {0}")]
    InitializationFailed(String),
    #[error("Failed to create surface: {0}")]
    SurfaceCreationFailed(String),
    #[error("Failed to create device: {0}")]
    DeviceCreationFailed(String),
    #[error("Failed to create swapchain: {0}")]
    SwapchainCreationFailed(String),
    #[error("Failed to acquire next image: {0}")]
    AcquireImageFailed(String),
    #[error("Failed to present: {0}")]
    PresentFailed(String),
    #[error("Failed to create buffer: {0}")]
    BufferCreationFailed(String),
    #[error("Failed to create texture: {0}")]
    TextureCreationFailed(String),
    #[error("Failed to create pipeline: {0}")]
    PipelineCreationFailed(String),
    #[error("Failed to create shader: {0}")]
    ShaderCreationFailed(String),
    #[error("Surface lost")]
    SurfaceLost,
    #[error("Out of memory")]
    OutOfMemory,
    #[error("Device lost")]
    DeviceLost,
}

pub type BackendResult<T> = Result<T, BackendError>;

/// Handle to a GPU buffer
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferHandle(pub(crate) u64);

/// Handle to a GPU texture
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureHandle(pub(crate) u64);

/// Handle to a texture view
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureViewHandle(pub(crate) u64);

/// Handle to a sampler
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SamplerHandle(pub(crate) u64);

/// Handle to a render pipeline
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RenderPipelineHandle(pub(crate) u64);

/// Handle to a compute pipeline
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ComputePipelineHandle(pub(crate) u64);

/// Handle to a bind group
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BindGroupHandle(pub(crate) u64);

/// Handle to a bind group layout
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BindGroupLayoutHandle(pub(crate) u64);

/// Bind group entry for creating bind groups
#[derive(Debug, Clone)]
pub enum BindGroupEntry {
    Buffer {
        buffer: BufferHandle,
        offset: u64,
        size: Option<u64>,
    },
    Texture(TextureViewHandle),
    Sampler(SamplerHandle),
    StorageTexture(TextureViewHandle),
}

/// Bind group layout entry
#[derive(Debug, Clone)]
pub struct BindGroupLayoutEntry {
    pub binding: u32,
    pub visibility: ShaderStageFlags,
    pub ty: BindingType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShaderStageFlags(u32);

impl ShaderStageFlags {
    pub const VERTEX: Self = Self(1 << 0);
    pub const FRAGMENT: Self = Self(1 << 1);
    pub const COMPUTE: Self = Self(1 << 2);
    pub const VERTEX_FRAGMENT: Self = Self((1 << 0) | (1 << 1));
    pub const ALL: Self = Self((1 << 0) | (1 << 1) | (1 << 2));

    pub fn contains(&self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
}

impl std::ops::BitOr for ShaderStageFlags {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

/// Binding type
#[derive(Debug, Clone)]
pub enum BindingType {
    UniformBuffer,
    StorageBuffer { read_only: bool },
    Texture { sample_type: TextureSampleType },
    StorageTexture { format: TextureFormat },
    Sampler { comparison: bool },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureSampleType {
    Float { filterable: bool },
    Depth,
    Sint,
    Uint,
}

/// Render pipeline descriptor
#[derive(Debug, Clone)]
pub struct RenderPipelineDescriptor {
    pub label: Option<String>,
    pub vertex_shader: String,
    pub fragment_shader: Option<String>,
    pub vertex_layouts: Vec<VertexBufferLayout>,
    pub bind_group_layouts: Vec<BindGroupLayoutHandle>,
    pub primitive_topology: PrimitiveTopology,
    pub front_face: FrontFace,
    pub cull_mode: CullMode,
    pub depth_stencil: Option<DepthStencilState>,
    pub color_targets: Vec<ColorTargetState>,
}

#[derive(Debug, Clone)]
pub struct DepthStencilState {
    pub format: TextureFormat,
    pub depth_write_enabled: bool,
    pub depth_compare: CompareFunction,
}

#[derive(Debug, Clone)]
pub struct ColorTargetState {
    pub format: TextureFormat,
    pub blend: Option<BlendState>,
    pub write_mask: ColorWrites,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColorWrites(pub u32);

impl ColorWrites {
    pub const RED: Self = Self(1 << 0);
    pub const GREEN: Self = Self(1 << 1);
    pub const BLUE: Self = Self(1 << 2);
    pub const ALPHA: Self = Self(1 << 3);
    pub const ALL: Self = Self(0xF);

    pub fn bits(&self) -> u32 {
        self.0
    }
}

/// Compute pipeline descriptor
#[derive(Debug, Clone)]
pub struct ComputePipelineDescriptor {
    pub label: Option<String>,
    pub shader: String,
    pub entry_point: String,
    pub bind_group_layouts: Vec<BindGroupLayoutHandle>,
}

/// Color attachment for render pass
#[derive(Debug, Clone)]
pub struct ColorAttachment {
    pub view: TextureViewHandle,
    pub resolve_target: Option<TextureViewHandle>,
    pub load_op: LoadOp,
    pub store_op: StoreOp,
}

#[derive(Debug, Clone)]
pub enum LoadOp {
    Clear([f32; 4]),
    Load,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoreOp {
    Store,
    Discard,
}

/// Depth stencil attachment for render pass
#[derive(Debug, Clone)]
pub struct DepthStencilAttachment {
    pub view: TextureViewHandle,
    pub depth_load_op: LoadOp,
    pub depth_store_op: StoreOp,
    pub depth_clear_value: f32,
}

/// Render pass descriptor
#[derive(Debug, Clone)]
pub struct RenderPassDescriptor {
    pub label: Option<String>,
    pub color_attachments: Vec<ColorAttachment>,
    pub depth_stencil_attachment: Option<DepthStencilAttachment>,
}

/// Frame context returned when beginning a frame
pub struct FrameContext {
    pub swapchain_view: TextureViewHandle,
    pub width: u32,
    pub height: u32,
}

/// Main graphics backend trait
pub trait GraphicsBackend: Sized {
    /// Create a new backend instance
    fn new(window: Arc<winit::window::Window>, vsync: bool) -> BackendResult<Self>;

    /// Resize the swapchain
    fn resize(&mut self, width: u32, height: u32);

    /// Get the actual surface size (may be clamped by device limits)
    fn surface_size(&self) -> (u32, u32);

    /// Begin a new frame
    fn begin_frame(&mut self) -> BackendResult<FrameContext>;

    /// End and present the frame
    fn end_frame(&mut self) -> BackendResult<()>;

    /// Get the swapchain format
    fn swapchain_format(&self) -> TextureFormat;

    // Resource creation

    /// Create a buffer
    fn create_buffer(&mut self, desc: &BufferDescriptor) -> BackendResult<BufferHandle>;

    /// Create a buffer with initial data
    fn create_buffer_init(&mut self, desc: &BufferDescriptor, data: &[u8])
        -> BackendResult<BufferHandle>;

    /// Write data to a buffer
    fn write_buffer(&mut self, buffer: BufferHandle, offset: u64, data: &[u8]);

    /// Create a texture
    fn create_texture(&mut self, desc: &TextureDescriptor) -> BackendResult<TextureHandle>;

    /// Create a texture view
    fn create_texture_view(&mut self, texture: TextureHandle) -> BackendResult<TextureViewHandle>;

    /// Write data to a texture
    fn write_texture(&mut self, texture: TextureHandle, data: &[u8], width: u32, height: u32);

    /// Create a sampler
    fn create_sampler(&mut self, desc: &SamplerDescriptor) -> BackendResult<SamplerHandle>;

    // Pipeline creation

    /// Create a bind group layout
    fn create_bind_group_layout(
        &mut self,
        entries: &[BindGroupLayoutEntry],
    ) -> BackendResult<BindGroupLayoutHandle>;

    /// Create a bind group
    fn create_bind_group(
        &mut self,
        layout: BindGroupLayoutHandle,
        entries: &[(u32, BindGroupEntry)],
    ) -> BackendResult<BindGroupHandle>;

    /// Create a render pipeline
    fn create_render_pipeline(
        &mut self,
        desc: &RenderPipelineDescriptor,
    ) -> BackendResult<RenderPipelineHandle>;

    /// Create a compute pipeline
    fn create_compute_pipeline(
        &mut self,
        desc: &ComputePipelineDescriptor,
    ) -> BackendResult<ComputePipelineHandle>;

    // Command recording and execution

    /// Begin a render pass
    fn begin_render_pass(&mut self, desc: &RenderPassDescriptor);

    /// End the current render pass
    fn end_render_pass(&mut self);

    /// Begin a compute pass
    fn begin_compute_pass(&mut self, label: Option<&str>);

    /// End the current compute pass
    fn end_compute_pass(&mut self);

    /// Set the render pipeline
    fn set_render_pipeline(&mut self, pipeline: RenderPipelineHandle);

    /// Set the compute pipeline
    fn set_compute_pipeline(&mut self, pipeline: ComputePipelineHandle);

    /// Set a bind group
    fn set_bind_group(&mut self, index: u32, bind_group: BindGroupHandle);

    /// Set vertex buffer
    fn set_vertex_buffer(&mut self, slot: u32, buffer: BufferHandle, offset: u64);

    /// Set index buffer
    fn set_index_buffer(&mut self, buffer: BufferHandle, offset: u64, format: IndexFormat);

    /// Set viewport
    fn set_viewport(&mut self, x: f32, y: f32, width: f32, height: f32, min_depth: f32, max_depth: f32);

    /// Set scissor rect
    fn set_scissor_rect(&mut self, x: u32, y: u32, width: u32, height: u32);

    /// Draw primitives
    fn draw(&mut self, vertices: std::ops::Range<u32>, instances: std::ops::Range<u32>);

    /// Draw indexed primitives
    fn draw_indexed(
        &mut self,
        indices: std::ops::Range<u32>,
        base_vertex: i32,
        instances: std::ops::Range<u32>,
    );

    /// Dispatch compute work
    fn dispatch_compute(&mut self, x: u32, y: u32, z: u32);

    // Resource cleanup

    /// Destroy a buffer
    fn destroy_buffer(&mut self, buffer: BufferHandle);

    /// Destroy a texture
    fn destroy_texture(&mut self, texture: TextureHandle);
}

/// Index format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexFormat {
    Uint16,
    Uint32,
}
