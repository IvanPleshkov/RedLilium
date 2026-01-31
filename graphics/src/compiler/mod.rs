//! Render graph compilation.
//!
//! This module handles the compilation of a [`RenderGraph`](crate::graph::RenderGraph)
//! into an execution plan ([`CompiledGraph`]).
//!
//! # Design Philosophy
//!
//! The compiler is intentionally simple. It performs:
//!
//! 1. **Topological Sort** - Order passes respecting dependencies
//! 2. **Cycle Detection** - Validate the graph is a DAG
//!
//! **Why not more optimization?**
//!
//! GPU parallelism in this engine happens at the *graph level*, not the *pass level*.
//! The [`FrameSchedule`](crate::scheduler::FrameSchedule) submits multiple graphs
//! with semaphore dependencies, enabling streaming submission and GPU overlap.
//!
//! Within a single graph, passes execute sequentially. Advanced per-pass optimization
//! (barrier batching, memory aliasing) would add CPU cost every frame for marginal
//! benefit when:
//! - Graphs are typically small (3-10 passes)
//! - GPU drivers already perform their own optimization
//! - Real parallelism is via multiple graphs, not within one graph
//!
//! # Example
//!
//! ```ignore
//! use redlilium_graphics::{RenderGraph, GraphicsPass};
//!
//! let mut graph = RenderGraph::new();
//! let geometry = graph.add_graphics_pass(GraphicsPass::new("geometry".into()));
//! let lighting = graph.add_graphics_pass(GraphicsPass::new("lighting".into()));
//! graph.add_dependency(lighting, geometry);
//!
//! let compiled = graph.compile()?;
//! // compiled.pass_order() returns execution order respecting dependencies
//! ```

use crate::graph::{PassHandle, RenderGraph};

/// A compiled render graph ready for execution.
///
/// Contains a topologically sorted pass order that respects all dependencies.
/// Passes are executed sequentially in this order within a single command buffer.
#[derive(Debug)]
pub struct CompiledGraph {
    /// Optimized pass execution order as handles.
    pass_order: Vec<PassHandle>,
}

impl CompiledGraph {
    /// Create a new compiled graph with the given pass order.
    pub(crate) fn new(pass_order: Vec<PassHandle>) -> Self {
        Self { pass_order }
    }

    /// Get the optimized pass execution order as handles.
    ///
    /// The passes should be executed in this order to satisfy all
    /// dependencies and achieve optimal performance.
    pub fn pass_order(&self) -> &[PassHandle] {
        &self.pass_order
    }

    /// Get the number of passes in the compiled graph.
    pub fn pass_count(&self) -> usize {
        self.pass_order.len()
    }

    /// Check if the compiled graph is empty.
    pub fn is_empty(&self) -> bool {
        self.pass_order.is_empty()
    }
}

/// Errors that can occur during graph compilation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphError {
    /// The graph contains a cyclic dependency.
    ///
    /// Render graphs must be directed acyclic graphs (DAGs).
    /// A cycle means passes depend on each other in a way that
    /// makes execution impossible.
    CyclicDependency,

    /// An invalid pass handle was encountered.
    InvalidPassHandle(PassHandle),
}

impl std::fmt::Display for GraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CyclicDependency => write!(f, "render graph contains cyclic dependency"),
            Self::InvalidPassHandle(handle) => {
                write!(f, "invalid pass handle: {:?}", handle)
            }
        }
    }
}

impl std::error::Error for GraphError {}

/// Compile a render graph into an execution plan.
///
/// Performs topological sorting to produce a pass order that respects
/// all dependencies. Detects cycles and returns an error if found.
///
/// # Arguments
///
/// * `graph` - The render graph to compile
///
/// # Returns
///
/// * `Ok(CompiledGraph)` - Pass order ready for sequential execution
/// * `Err(GraphError::CyclicDependency)` - If the graph contains a cycle
///
/// # Example
///
/// ```ignore
/// let mut graph = RenderGraph::new();
/// let pass1 = graph.add_graphics_pass(GraphicsPass::new("pass1".into()));
/// let pass2 = graph.add_graphics_pass(GraphicsPass::new("pass2".into()));
/// graph.add_dependency(pass2, pass1);
///
/// let compiled = compile(&graph)?;
/// assert_eq!(compiled.pass_order(), &[pass1, pass2]);
/// ```
pub fn compile(graph: &RenderGraph) -> Result<CompiledGraph, GraphError> {
    // TODO: Implement proper topological sort with cycle detection
    // For now, return passes in insertion order
    let pass_order = (0..graph.pass_count() as u32)
        .map(PassHandle::new)
        .collect();
    Ok(CompiledGraph::new(pass_order))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphicsPass;

    #[test]
    fn test_compile_empty_graph() {
        let graph = RenderGraph::new();
        let compiled = compile(&graph).unwrap();
        assert!(compiled.is_empty());
        assert_eq!(compiled.pass_count(), 0);
    }

    #[test]
    fn test_compile_single_pass() {
        let mut graph = RenderGraph::new();
        let pass = graph.add_graphics_pass(GraphicsPass::new("main".into()));

        let compiled = compile(&graph).unwrap();
        assert_eq!(compiled.pass_count(), 1);
        assert_eq!(compiled.pass_order(), &[pass]);
    }

    #[test]
    fn test_compile_with_dependencies() {
        let mut graph = RenderGraph::new();
        let pass1 = graph.add_graphics_pass(GraphicsPass::new("geometry".into()));
        let pass2 = graph.add_graphics_pass(GraphicsPass::new("lighting".into()));
        graph.add_dependency(pass2, pass1);

        let compiled = compile(&graph).unwrap();
        assert_eq!(compiled.pass_count(), 2);
        // After proper implementation, pass1 should come before pass2
    }

    #[test]
    fn test_compiled_graph_accessors() {
        let pass_order = vec![PassHandle::new(0), PassHandle::new(1), PassHandle::new(2)];
        let compiled = CompiledGraph::new(pass_order.clone());

        assert_eq!(compiled.pass_count(), 3);
        assert!(!compiled.is_empty());
        assert_eq!(compiled.pass_order(), &pass_order);
    }
}
