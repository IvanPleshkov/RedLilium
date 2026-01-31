//! Dummy backend for testing.
//!
//! This backend performs no GPU operations and is used for:
//! - Unit testing without GPU hardware
//! - CI environments without graphics support
//! - Validating render graph construction

use super::{Backend, BackendError};
use crate::graph::CompiledGraph;
use crate::types::{BufferDescriptor, SamplerDescriptor, TextureDescriptor};

/// A dummy buffer (no actual GPU resource).
#[derive(Debug)]
pub struct DummyBuffer {
    /// Size in bytes.
    pub size: u64,
    /// Debug label.
    pub label: Option<String>,
}

/// A dummy texture (no actual GPU resource).
#[derive(Debug)]
pub struct DummyTexture {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Debug label.
    pub label: Option<String>,
}

/// A dummy sampler (no actual GPU resource).
#[derive(Debug)]
pub struct DummySampler {
    /// Debug label.
    pub label: Option<String>,
}

/// A dummy command buffer.
#[derive(Debug, Default)]
pub struct DummyCommandBuffer {
    /// Number of recorded commands.
    pub command_count: u32,
}

/// No-op graphics backend for testing.
///
/// This backend validates API usage without performing any GPU operations.
/// All resource creation succeeds and all commands are no-ops.
///
/// # Example
///
/// ```
/// use redlilium_graphics::{DummyBackend, Backend};
///
/// let backend = DummyBackend::new();
/// assert_eq!(backend.name(), "Dummy");
/// ```
#[derive(Debug, Default)]
pub struct DummyBackend {
    /// Track number of resources created for debugging.
    resource_count: std::sync::atomic::AtomicU32,
}

impl DummyBackend {
    /// Create a new dummy backend.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the number of resources created.
    pub fn resource_count(&self) -> u32 {
        self.resource_count
            .load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl Backend for DummyBackend {
    type Buffer = DummyBuffer;
    type Texture = DummyTexture;
    type Sampler = DummySampler;
    type CommandBuffer = DummyCommandBuffer;

    fn name(&self) -> &'static str {
        "Dummy"
    }

    fn create_buffer(&self, descriptor: &BufferDescriptor) -> Result<Self::Buffer, BackendError> {
        self.resource_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        log::trace!("DummyBackend: creating buffer {:?}", descriptor.label);
        Ok(DummyBuffer {
            size: descriptor.size,
            label: descriptor.label.clone(),
        })
    }

    fn destroy_buffer(&self, _buffer: Self::Buffer) {
        log::trace!("DummyBackend: destroying buffer");
    }

    fn create_texture(
        &self,
        descriptor: &TextureDescriptor,
    ) -> Result<Self::Texture, BackendError> {
        self.resource_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        log::trace!("DummyBackend: creating texture {:?}", descriptor.label);
        Ok(DummyTexture {
            width: descriptor.size.width,
            height: descriptor.size.height,
            label: descriptor.label.clone(),
        })
    }

    fn destroy_texture(&self, _texture: Self::Texture) {
        log::trace!("DummyBackend: destroying texture");
    }

    fn create_sampler(
        &self,
        descriptor: &SamplerDescriptor,
    ) -> Result<Self::Sampler, BackendError> {
        self.resource_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        log::trace!("DummyBackend: creating sampler {:?}", descriptor.label);
        Ok(DummySampler {
            label: descriptor.label.clone(),
        })
    }

    fn destroy_sampler(&self, _sampler: Self::Sampler) {
        log::trace!("DummyBackend: destroying sampler");
    }

    fn begin_command_buffer(&self) -> Result<Self::CommandBuffer, BackendError> {
        log::trace!("DummyBackend: beginning command buffer");
        Ok(DummyCommandBuffer::default())
    }

    fn submit_command_buffer(
        &self,
        command_buffer: Self::CommandBuffer,
    ) -> Result<(), BackendError> {
        log::trace!(
            "DummyBackend: submitting command buffer with {} commands",
            command_buffer.command_count
        );
        Ok(())
    }

    fn execute_graph(&self, graph: &CompiledGraph) -> Result<(), BackendError> {
        log::trace!(
            "DummyBackend: executing graph with {} passes",
            graph.pass_order().len()
        );
        Ok(())
    }

    fn wait_idle(&self) -> Result<(), BackendError> {
        log::trace!("DummyBackend: wait_idle (no-op)");
        Ok(())
    }
}

// Ensure DummyBackend is Send + Sync
static_assertions::assert_impl_all!(DummyBackend: Send, Sync);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{BufferUsage, TextureFormat, TextureUsage};

    #[test]
    fn test_dummy_backend_name() {
        let backend = DummyBackend::new();
        assert_eq!(backend.name(), "Dummy");
    }

    #[test]
    fn test_create_buffer() {
        let backend = DummyBackend::new();
        let descriptor = BufferDescriptor::new(1024, BufferUsage::VERTEX);
        let buffer = backend.create_buffer(&descriptor).unwrap();
        assert_eq!(buffer.size, 1024);
    }

    #[test]
    fn test_create_texture() {
        let backend = DummyBackend::new();
        let descriptor = TextureDescriptor::new_2d(
            512,
            512,
            TextureFormat::Rgba8Unorm,
            TextureUsage::TEXTURE_BINDING,
        );
        let texture = backend.create_texture(&descriptor).unwrap();
        assert_eq!(texture.width, 512);
        assert_eq!(texture.height, 512);
    }

    #[test]
    fn test_resource_counting() {
        let backend = DummyBackend::new();
        assert_eq!(backend.resource_count(), 0);

        let _ = backend.create_buffer(&BufferDescriptor::new(64, BufferUsage::UNIFORM));
        assert_eq!(backend.resource_count(), 1);

        let _ = backend.create_texture(&TextureDescriptor::new_2d(
            256,
            256,
            TextureFormat::Rgba8Unorm,
            TextureUsage::RENDER_ATTACHMENT,
        ));
        assert_eq!(backend.resource_count(), 2);
    }
}
