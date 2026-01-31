//! Graphics backend implementations.
//!
//! This module provides the [`Backend`] trait and implementations for
//! different graphics APIs:
//!
//! - [`DummyBackend`] - No-op backend for testing
//! - Vulkan backend (planned) - High-performance via ash crate
//! - wgpu backend (planned) - Cross-platform via wgpu 28.0.0

mod dummy;
mod error;

pub use dummy::DummyBackend;
pub use error::BackendError;

use crate::graph::CompiledGraph;
use crate::types::{BufferDescriptor, SamplerDescriptor, TextureDescriptor};

/// Trait for graphics backend implementations.
///
/// Backends provide the low-level graphics API operations that the
/// render graph executor uses to run rendering commands.
///
/// # Thread Safety
///
/// All backends must be `Send + Sync` to support multithreaded
/// command recording. Specific thread-safety guarantees vary by
/// implementation.
///
/// # Implementations
///
/// - [`DummyBackend`]: No-op implementation for testing
/// - `VulkanBackend`: (planned) Direct Vulkan via ash
/// - `WgpuBackend`: (planned) Cross-platform via wgpu
pub trait Backend: Send + Sync {
    /// Type representing a GPU buffer.
    type Buffer: Send + Sync;
    /// Type representing a GPU texture.
    type Texture: Send + Sync;
    /// Type representing a texture sampler.
    type Sampler: Send + Sync;
    /// Type representing a command buffer.
    type CommandBuffer: Send + Sync;

    /// Get the backend name for debugging.
    fn name(&self) -> &'static str;

    /// Create a GPU buffer.
    fn create_buffer(&self, descriptor: &BufferDescriptor) -> Result<Self::Buffer, BackendError>;

    /// Destroy a GPU buffer.
    fn destroy_buffer(&self, buffer: Self::Buffer);

    /// Create a GPU texture.
    fn create_texture(&self, descriptor: &TextureDescriptor)
    -> Result<Self::Texture, BackendError>;

    /// Destroy a GPU texture.
    fn destroy_texture(&self, texture: Self::Texture);

    /// Create a texture sampler.
    fn create_sampler(&self, descriptor: &SamplerDescriptor)
    -> Result<Self::Sampler, BackendError>;

    /// Destroy a texture sampler.
    fn destroy_sampler(&self, sampler: Self::Sampler);

    /// Begin a new command buffer for recording.
    fn begin_command_buffer(&self) -> Result<Self::CommandBuffer, BackendError>;

    /// End and submit a command buffer.
    fn submit_command_buffer(
        &self,
        command_buffer: Self::CommandBuffer,
    ) -> Result<(), BackendError>;

    /// Execute a compiled render graph.
    fn execute_graph(&self, graph: &CompiledGraph) -> Result<(), BackendError>;

    /// Wait for all GPU work to complete.
    fn wait_idle(&self) -> Result<(), BackendError>;
}

/// Capability flags for backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BackendCapabilities {
    /// Maximum texture dimension.
    pub max_texture_dimension: u32,
    /// Maximum buffer size.
    pub max_buffer_size: u64,
    /// Whether compute shaders are supported.
    pub compute_shaders: bool,
    /// Whether ray tracing is supported.
    pub ray_tracing: bool,
    /// Whether mesh shaders are supported.
    pub mesh_shaders: bool,
}

impl Default for BackendCapabilities {
    fn default() -> Self {
        Self {
            max_texture_dimension: 16384,
            max_buffer_size: 1 << 30, // 1 GB
            compute_shaders: true,
            ray_tracing: false,
            mesh_shaders: false,
        }
    }
}
