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

use crate::types::{BufferDescriptor, TextureDescriptor};

/// Handle to a texture resource in the render graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureHandle(ResourceHandle);

/// Handle to a buffer resource in the render graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferHandle(ResourceHandle);

/// The render graph describes a frame's rendering operations.
///
/// # Construction
///
/// Build a graph by creating resources and adding passes:
///
/// ```ignore
/// let mut graph = RenderGraph::new();
///
/// let depth = graph.create_texture(TextureDescriptor::new_2d(
///     1920, 1080,
///     TextureFormat::Depth32Float,
///     TextureUsage::RENDER_ATTACHMENT,
/// ));
///
/// graph.add_pass("geometry", |builder| {
///     builder.write_depth(depth);
/// });
/// ```
///
/// # Execution
///
/// After construction, the graph is compiled and executed with a backend:
///
/// ```ignore
/// graph.execute(&backend)?;
/// ```
#[derive(Debug, Default)]
pub struct RenderGraph {
    /// All passes in the graph.
    passes: Vec<RenderPass>,
    /// Texture descriptors.
    textures: Vec<TextureDescriptor>,
    /// Buffer descriptors.
    buffers: Vec<BufferDescriptor>,
}

impl RenderGraph {
    /// Create a new empty render graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a texture resource in the graph.
    ///
    /// The texture is not allocated until the graph is executed.
    pub fn create_texture(&mut self, descriptor: TextureDescriptor) -> TextureHandle {
        let index = self.textures.len();
        self.textures.push(descriptor);
        TextureHandle(ResourceHandle::new(index as u32))
    }

    /// Create a buffer resource in the graph.
    ///
    /// The buffer is not allocated until the graph is executed.
    pub fn create_buffer(&mut self, descriptor: BufferDescriptor) -> BufferHandle {
        let index = self.buffers.len();
        self.buffers.push(descriptor);
        BufferHandle(ResourceHandle::new(index as u32))
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

    /// Get all texture descriptors.
    pub fn textures(&self) -> &[TextureDescriptor] {
        &self.textures
    }

    /// Get all buffer descriptors.
    pub fn buffers(&self) -> &[BufferDescriptor] {
        &self.buffers
    }

    /// Get the number of passes in the graph.
    pub fn pass_count(&self) -> usize {
        self.passes.len()
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
    use crate::types::{TextureFormat, TextureUsage};

    #[test]
    fn test_create_texture() {
        let mut graph = RenderGraph::new();
        let _handle = graph.create_texture(TextureDescriptor::new_2d(
            1920,
            1080,
            TextureFormat::Rgba8Unorm,
            TextureUsage::RENDER_ATTACHMENT,
        ));
        assert_eq!(graph.textures().len(), 1);
    }

    #[test]
    fn test_add_pass() {
        let mut graph = RenderGraph::new();
        let _handle = graph.add_pass("test_pass", PassType::Graphics);
        assert_eq!(graph.pass_count(), 1);
    }
}
