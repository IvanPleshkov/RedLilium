//! Render graph infrastructure.
//!
//! The render graph provides a declarative way to describe rendering operations
//! and their dependencies. The graph compiler automatically handles:
//!
//! - Optimal pass ordering via topological sort
//! - Resource lifetime analysis
//! - Synchronization and barrier insertion
//! - Memory aliasing opportunities
//!
//! # Render Targets
//!
//! Graphics passes can have render targets configured to specify where they render:
//!
//! ```ignore
//! use redlilium_graphics::{RenderGraph, PassType, ColorAttachment, RenderTargetConfig, LoadOp};
//!
//! let mut graph = RenderGraph::new();
//! let pass = graph.add_pass("main", PassType::Graphics);
//!
//! // Configure to render to surface
//! graph.set_render_targets(pass, RenderTargetConfig::new()
//!     .with_color(ColorAttachment::from_surface(&surface_texture)
//!         .with_clear_color(0.0, 0.0, 0.0, 1.0)));
//! ```

mod pass;
mod resource;
mod target;

pub use pass::{PassHandle, PassType, RenderPass};
pub use resource::ResourceHandle;
pub use target::{
    ColorAttachment, DepthStencilAttachment, LoadOp, RenderTarget, RenderTargetConfig, StoreOp,
};

/// The render graph describes a frame's rendering operations.
///
/// # Construction
///
/// Build a graph by adding passes:
///
/// ```ignore
/// let mut graph = RenderGraph::new();
/// graph.add_pass("geometry", PassType::Graphics);
/// graph.add_pass("lighting", PassType::Graphics);
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
}

impl RenderGraph {
    /// Create a new empty render graph.
    pub fn new() -> Self {
        Self::default()
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

    /// Get the number of passes in the graph.
    pub fn pass_count(&self) -> usize {
        self.passes.len()
    }

    /// Get a mutable reference to a pass by handle.
    pub fn get_pass_mut(&mut self, handle: PassHandle) -> Option<&mut RenderPass> {
        self.passes.get_mut(handle.index() as usize)
    }

    /// Get a reference to a pass by handle.
    pub fn get_pass(&self, handle: PassHandle) -> Option<&RenderPass> {
        self.passes.get(handle.index() as usize)
    }

    /// Set render targets for a graphics pass.
    ///
    /// # Errors
    ///
    /// Returns an error if the pass handle is invalid.
    pub fn set_render_targets(
        &mut self,
        handle: PassHandle,
        config: RenderTargetConfig,
    ) -> Result<(), GraphError> {
        let pass = self
            .passes
            .get_mut(handle.index() as usize)
            .ok_or(GraphError::InvalidPass)?;
        pass.set_render_targets(config);
        Ok(())
    }

    /// Add a dependency between passes.
    ///
    /// The dependent pass will execute after the dependency.
    ///
    /// # Errors
    ///
    /// Returns an error if either pass handle is invalid.
    pub fn add_dependency(
        &mut self,
        dependent: PassHandle,
        dependency: PassHandle,
    ) -> Result<(), GraphError> {
        // Validate both handles exist
        if self.passes.get(dependency.index() as usize).is_none() {
            return Err(GraphError::InvalidPass);
        }
        let pass = self
            .passes
            .get_mut(dependent.index() as usize)
            .ok_or(GraphError::InvalidPass)?;
        pass.add_dependency(dependency);
        Ok(())
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

    /// Clear all passes from the graph.
    pub fn clear(&mut self) {
        self.passes.clear();
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
    use crate::types::{TextureDescriptor, TextureFormat, TextureUsage};

    #[test]
    fn test_add_pass() {
        let mut graph = RenderGraph::new();
        let _handle = graph.add_pass("test_pass", PassType::Graphics);
        assert_eq!(graph.pass_count(), 1);
    }

    #[test]
    fn test_clear() {
        let mut graph = RenderGraph::new();
        graph.add_pass("test_pass", PassType::Graphics);

        graph.clear();

        assert_eq!(graph.pass_count(), 0);
    }

    #[test]
    fn test_set_render_targets() {
        let instance = GraphicsInstance::new().unwrap();
        let device = instance.create_device().unwrap();
        let texture = device
            .create_texture(&TextureDescriptor::new_2d(
                1920,
                1080,
                TextureFormat::Rgba8Unorm,
                TextureUsage::RENDER_ATTACHMENT,
            ))
            .unwrap();

        let mut graph = RenderGraph::new();
        let pass = graph.add_pass("main", PassType::Graphics);

        let config = RenderTargetConfig::new().with_color(
            ColorAttachment::from_texture(texture).with_clear_color(0.0, 0.0, 0.0, 1.0),
        );

        graph.set_render_targets(pass, config).unwrap();

        let render_pass = graph.get_pass(pass).unwrap();
        assert!(render_pass.has_render_targets());
    }

    #[test]
    fn test_add_dependency() {
        let mut graph = RenderGraph::new();
        let pass1 = graph.add_pass("geometry", PassType::Graphics);
        let pass2 = graph.add_pass("lighting", PassType::Graphics);

        graph.add_dependency(pass2, pass1).unwrap();

        let lighting_pass = graph.get_pass(pass2).unwrap();
        assert_eq!(lighting_pass.dependencies().len(), 1);
    }

    #[test]
    fn test_invalid_pass_handle() {
        let mut graph = RenderGraph::new();
        let pass = graph.add_pass("test", PassType::Graphics);

        // Clear the graph, making the handle invalid
        graph.clear();

        let result = graph.set_render_targets(pass, RenderTargetConfig::new());
        assert!(result.is_err());
    }
}
