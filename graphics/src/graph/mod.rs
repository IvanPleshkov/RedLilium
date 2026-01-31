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
}
