//! Render graph infrastructure.
//!
//! The render graph provides a declarative way to describe rendering operations
//! and their dependencies. The graph compiler automatically handles:
//!
//! - Optimal pass ordering via topological sort
//! - Resource lifetime analysis
//! - Synchronization and barrier insertion
//! - Memory aliasing opportunities

mod pass;
mod resource;

pub use pass::{PassHandle, PassType, RenderPass};
pub use resource::ResourceHandle;

use std::sync::Arc;

use crate::resources::{Buffer, Texture};

/// The render graph describes a frame's rendering operations.
///
/// # Construction
///
/// Build a graph by adding resources and passes:
///
/// ```ignore
/// let mut graph = RenderGraph::new();
///
/// let depth_texture = device.create_texture(&TextureDescriptor::new_2d(
///     1920, 1080,
///     TextureFormat::Depth32Float,
///     TextureUsage::RENDER_ATTACHMENT,
/// ))?;
///
/// graph.add_texture(depth_texture);
///
/// graph.add_pass("geometry", PassType::Graphics);
/// ```
///
/// # Execution
///
/// After construction, the graph is compiled and executed:
///
/// ```ignore
/// let compiled = graph.compile()?;
/// ```
#[derive(Debug, Default)]
pub struct RenderGraph {
    /// All passes in the graph.
    passes: Vec<RenderPass>,
    /// Textures used by the graph.
    textures: Vec<Arc<Texture>>,
    /// Buffers used by the graph.
    buffers: Vec<Arc<Buffer>>,
}

impl RenderGraph {
    /// Create a new empty render graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a texture resource to the graph.
    ///
    /// The texture must have been created by a [`GraphicsDevice`].
    ///
    /// [`GraphicsDevice`]: crate::GraphicsDevice
    pub fn add_texture(&mut self, texture: Arc<Texture>) {
        self.textures.push(texture);
    }

    /// Add a buffer resource to the graph.
    ///
    /// The buffer must have been created by a [`GraphicsDevice`].
    ///
    /// [`GraphicsDevice`]: crate::GraphicsDevice
    pub fn add_buffer(&mut self, buffer: Arc<Buffer>) {
        self.buffers.push(buffer);
    }

    /// Check if the graph contains a texture.
    pub fn contains_texture(&self, texture: &Arc<Texture>) -> bool {
        self.textures.iter().any(|t| Arc::ptr_eq(t, texture))
    }

    /// Check if the graph contains a buffer.
    pub fn contains_buffer(&self, buffer: &Arc<Buffer>) -> bool {
        self.buffers.iter().any(|b| Arc::ptr_eq(b, buffer))
    }

    /// Add a render pass to the graph.
    ///
    /// Returns a handle to the pass for dependency tracking.
    pub fn add_pass(&mut self, name: impl Into<String>, pass_type: PassType) -> PassHandle {
        let index = self.passes.len();
        self.passes.push(RenderPass::new(name.into(), pass_type));
        PassHandle::new(index as u32)
    }

    /// Get all passes in the graph.
    pub fn passes(&self) -> &[RenderPass] {
        &self.passes
    }

    /// Get all imported textures.
    pub fn textures(&self) -> &[Arc<Texture>] {
        &self.textures
    }

    /// Get all imported buffers.
    pub fn buffers(&self) -> &[Arc<Buffer>] {
        &self.buffers
    }

    /// Get the number of passes in the graph.
    pub fn pass_count(&self) -> usize {
        self.passes.len()
    }

    /// Get the number of imported textures.
    pub fn texture_count(&self) -> usize {
        self.textures.len()
    }

    /// Get the number of imported buffers.
    pub fn buffer_count(&self) -> usize {
        self.buffers.len()
    }

    /// Compile the graph for execution.
    ///
    /// This performs:
    /// - Topological sorting of passes
    /// - Resource lifetime analysis
    /// - Barrier placement optimization
    pub fn compile(&mut self) -> Result<CompiledGraph, GraphError> {
        // TODO: Implement graph compilation
        Ok(CompiledGraph {
            pass_order: (0..self.passes.len()).collect(),
        })
    }

    /// Clear all passes and resources from the graph.
    pub fn clear(&mut self) {
        self.passes.clear();
        self.textures.clear();
        self.buffers.clear();
    }
}

/// A compiled render graph ready for execution.
#[derive(Debug)]
pub struct CompiledGraph {
    /// Optimized pass execution order.
    pass_order: Vec<usize>,
}

impl CompiledGraph {
    /// Get the optimized pass execution order.
    pub fn pass_order(&self) -> &[usize] {
        &self.pass_order
    }
}

/// Errors that can occur during graph construction or compilation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphError {
    /// The graph contains a cycle.
    CyclicDependency,
    /// A resource handle is invalid.
    InvalidResource,
    /// A pass handle is invalid.
    InvalidPass,
}

impl std::fmt::Display for GraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CyclicDependency => write!(f, "render graph contains cyclic dependency"),
            Self::InvalidResource => write!(f, "invalid resource handle"),
            Self::InvalidPass => write!(f, "invalid pass handle"),
        }
    }
}

impl std::error::Error for GraphError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instance::GraphicsInstance;
    use crate::types::{
        BufferDescriptor, BufferUsage, TextureDescriptor, TextureFormat, TextureUsage,
    };

    fn create_test_device() -> Arc<crate::device::GraphicsDevice> {
        let instance = GraphicsInstance::new().unwrap();
        instance.create_device().unwrap()
    }

    fn create_test_texture() -> Arc<Texture> {
        let device = create_test_device();
        device
            .create_texture(&TextureDescriptor::new_2d(
                1920,
                1080,
                TextureFormat::Rgba8Unorm,
                TextureUsage::RENDER_ATTACHMENT,
            ))
            .unwrap()
    }

    fn create_test_buffer() -> Arc<Buffer> {
        let device = create_test_device();
        device
            .create_buffer(&BufferDescriptor::new(1024, BufferUsage::VERTEX))
            .unwrap()
    }

    #[test]
    fn test_add_texture() {
        let mut graph = RenderGraph::new();
        let texture = create_test_texture();
        graph.add_texture(texture.clone());
        assert_eq!(graph.texture_count(), 1);
        assert!(graph.contains_texture(&texture));
    }

    #[test]
    fn test_add_buffer() {
        let mut graph = RenderGraph::new();
        let buffer = create_test_buffer();
        graph.add_buffer(buffer.clone());
        assert_eq!(graph.buffer_count(), 1);
        assert!(graph.contains_buffer(&buffer));
    }

    #[test]
    fn test_add_pass() {
        let mut graph = RenderGraph::new();
        let _handle = graph.add_pass("test_pass", PassType::Graphics);
        assert_eq!(graph.pass_count(), 1);
    }

    #[test]
    fn test_clear() {
        let mut graph = RenderGraph::new();
        graph.add_texture(create_test_texture());
        graph.add_buffer(create_test_buffer());
        graph.add_pass("test_pass", PassType::Graphics);

        graph.clear();

        assert_eq!(graph.texture_count(), 0);
        assert_eq!(graph.buffer_count(), 0);
        assert_eq!(graph.pass_count(), 0);
    }
}
