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
//! use redlilium_graphics::GraphicsPass;
//!
//! let mut graph = schedule.acquire_graph();
//! let geometry = graph.add_graphics_pass(GraphicsPass::new("geometry".into()));
//! let lighting = graph.add_graphics_pass(GraphicsPass::new("lighting".into()));
//! graph.add_dependency(lighting, geometry);
//!
//! let compiled = graph.compile()?;
//! // compiled.pass_order() returns execution order respecting dependencies
//! ```

use std::collections::VecDeque;

use redlilium_core::pool::Poolable;

use crate::graph::{Pass, PassHandle, RenderGraph};

/// A compiled render graph ready for execution.
///
/// Contains a topologically sorted pass order that respects all dependencies.
/// Passes are executed sequentially in this order within a single command buffer.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct CompiledGraph {
    /// Optimized pass execution order as handles.
    pass_order: Vec<PassHandle>,
}

impl CompiledGraph {
    /// Create a new compiled graph with the given pass order.
    #[cfg(test)]
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

impl Poolable for CompiledGraph {
    fn reset(&mut self) {
        self.pass_order.clear();
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
    let mut result = CompiledGraph::default();
    compile_into(graph.passes(), graph.edges(), &mut result)?;
    Ok(result)
}

/// Compile a render graph into an existing [`CompiledGraph`], reusing its allocation.
///
/// This is the in-place variant of [`compile`]. It clears the target and fills
/// it with the topologically sorted pass order. Used internally by
/// [`RenderGraph::compile`](crate::graph::RenderGraph::compile) to avoid
/// reallocating the pass order vector each frame.
pub(crate) fn compile_into(
    passes: &[Pass],
    edges: &[(PassHandle, PassHandle)],
    target: &mut CompiledGraph,
) -> Result<(), GraphError> {
    let n = passes.len();
    target.pass_order.clear();

    // Empty graph compiles to empty result
    if n == 0 {
        return Ok(());
    }

    // Kahn's algorithm for topological sort
    //
    // 1. Compute in-degree for each pass (number of dependencies)
    // 2. Start with passes that have no dependencies (in-degree = 0)
    // 3. Process them, reducing in-degree of passes that depend on them
    // 4. If we can't process all passes, there's a cycle

    // Compute in-degree for each pass
    // Edge (dependent, dependency) means dependent has one more in-degree
    let mut in_degree = vec![0u32; n];
    for &(dependent, _dependency) in edges {
        in_degree[dependent.index()] += 1;
    }

    // Queue passes with no dependencies
    let mut queue: VecDeque<PassHandle> = (0..n as u32)
        .map(PassHandle::new)
        .filter(|&h| in_degree[h.index()] == 0)
        .collect();

    while let Some(handle) = queue.pop_front() {
        target.pass_order.push(handle);

        // Find passes that depend on this one and reduce their in-degree
        for &(dependent, dependency) in edges {
            if dependency == handle {
                in_degree[dependent.index()] -= 1;
                if in_degree[dependent.index()] == 0 {
                    queue.push_back(dependent);
                }
            }
        }
    }

    // If we didn't process all passes, there's a cycle
    if target.pass_order.len() != n {
        target.pass_order.clear();
        return Err(GraphError::CyclicDependency);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{ComputePass, GraphicsPass, TransferPass};

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
    fn test_compile_linear_chain() {
        // A -> B -> C (A must come first, then B, then C)
        let mut graph = RenderGraph::new();
        let a = graph.add_graphics_pass(GraphicsPass::new("A".into()));
        let b = graph.add_graphics_pass(GraphicsPass::new("B".into()));
        let c = graph.add_graphics_pass(GraphicsPass::new("C".into()));

        graph.add_dependency(b, a); // B depends on A
        graph.add_dependency(c, b); // C depends on B

        let compiled = compile(&graph).unwrap();
        assert_eq!(compiled.pass_order(), &[a, b, c]);
    }

    #[test]
    fn test_compile_diamond_dependency() {
        //     A
        //    / \
        //   B   C
        //    \ /
        //     D
        let mut graph = RenderGraph::new();
        let a = graph.add_graphics_pass(GraphicsPass::new("A".into()));
        let b = graph.add_graphics_pass(GraphicsPass::new("B".into()));
        let c = graph.add_graphics_pass(GraphicsPass::new("C".into()));
        let d = graph.add_graphics_pass(GraphicsPass::new("D".into()));

        graph.add_dependency(b, a); // B depends on A
        graph.add_dependency(c, a); // C depends on A
        graph.add_dependency(d, b); // D depends on B
        graph.add_dependency(d, c); // D depends on C

        let compiled = compile(&graph).unwrap();
        let order = compiled.pass_order();

        // A must come first
        assert_eq!(order[0], a);
        // D must come last
        assert_eq!(order[3], d);
        // B and C must be in the middle (either order is valid)
        assert!(order[1] == b || order[1] == c);
        assert!(order[2] == b || order[2] == c);
        assert_ne!(order[1], order[2]);
    }

    #[test]
    fn test_compile_independent_passes() {
        // No dependencies - any order is valid, but all passes must be present
        let mut graph = RenderGraph::new();
        let a = graph.add_graphics_pass(GraphicsPass::new("A".into()));
        let b = graph.add_graphics_pass(GraphicsPass::new("B".into()));
        let c = graph.add_graphics_pass(GraphicsPass::new("C".into()));

        let compiled = compile(&graph).unwrap();
        let order = compiled.pass_order();

        assert_eq!(order.len(), 3);
        assert!(order.contains(&a));
        assert!(order.contains(&b));
        assert!(order.contains(&c));
    }

    #[test]
    fn test_compile_cycle_two_nodes() {
        // A -> B -> A (cycle)
        let mut graph = RenderGraph::new();
        let a = graph.add_graphics_pass(GraphicsPass::new("A".into()));
        let b = graph.add_graphics_pass(GraphicsPass::new("B".into()));

        graph.add_dependency(b, a); // B depends on A
        graph.add_dependency(a, b); // A depends on B (creates cycle)

        let result = compile(&graph);
        assert_eq!(result, Err(GraphError::CyclicDependency));
    }

    #[test]
    fn test_compile_cycle_three_nodes() {
        // A -> B -> C -> A (cycle)
        let mut graph = RenderGraph::new();
        let a = graph.add_graphics_pass(GraphicsPass::new("A".into()));
        let b = graph.add_graphics_pass(GraphicsPass::new("B".into()));
        let c = graph.add_graphics_pass(GraphicsPass::new("C".into()));

        graph.add_dependency(b, a); // B depends on A
        graph.add_dependency(c, b); // C depends on B
        graph.add_dependency(a, c); // A depends on C (creates cycle)

        let result = compile(&graph);
        assert_eq!(result, Err(GraphError::CyclicDependency));
    }

    #[test]
    fn test_compile_partial_cycle() {
        // D is independent, but A-B-C form a cycle
        //     D
        //   A -> B -> C -> A
        let mut graph = RenderGraph::new();
        let a = graph.add_graphics_pass(GraphicsPass::new("A".into()));
        let b = graph.add_graphics_pass(GraphicsPass::new("B".into()));
        let c = graph.add_graphics_pass(GraphicsPass::new("C".into()));
        let _d = graph.add_graphics_pass(GraphicsPass::new("D".into()));

        graph.add_dependency(b, a);
        graph.add_dependency(c, b);
        graph.add_dependency(a, c); // Creates cycle in A-B-C

        let result = compile(&graph);
        assert_eq!(result, Err(GraphError::CyclicDependency));
    }

    #[test]
    fn test_compile_mixed_pass_types() {
        // Transfer -> Compute -> Graphics
        let mut graph = RenderGraph::new();
        let upload = graph.add_transfer_pass(TransferPass::new("upload".into()));
        let simulate = graph.add_compute_pass(ComputePass::new("simulate".into()));
        let render = graph.add_graphics_pass(GraphicsPass::new("render".into()));

        graph.add_dependency(simulate, upload);
        graph.add_dependency(render, simulate);

        let compiled = compile(&graph).unwrap();
        assert_eq!(compiled.pass_order(), &[upload, simulate, render]);
    }

    #[test]
    fn test_compile_multiple_roots() {
        // Two independent chains: A->B and C->D
        let mut graph = RenderGraph::new();
        let a = graph.add_graphics_pass(GraphicsPass::new("A".into()));
        let b = graph.add_graphics_pass(GraphicsPass::new("B".into()));
        let c = graph.add_graphics_pass(GraphicsPass::new("C".into()));
        let d = graph.add_graphics_pass(GraphicsPass::new("D".into()));

        graph.add_dependency(b, a);
        graph.add_dependency(d, c);

        let compiled = compile(&graph).unwrap();
        let order = compiled.pass_order();

        // A must come before B
        assert!(order.iter().position(|&x| x == a) < order.iter().position(|&x| x == b));
        // C must come before D
        assert!(order.iter().position(|&x| x == c) < order.iter().position(|&x| x == d));
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
