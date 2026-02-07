//! Render graph compilation.
//!
//! This module handles the compilation of a [`RenderGraph`](crate::graph::RenderGraph)
//! into an execution plan ([`CompiledGraph`]).
//!
//! # Design Philosophy
//!
//! The compiler performs:
//!
//! 1. **Resource usage inference** - Analyze each pass's resource access
//! 2. **Auto-dependency generation** - Detect resource conflicts between passes
//! 3. **Topological sort** - Order passes respecting all dependencies
//! 4. **Cycle detection** - Validate the graph is a DAG
//!
//! The compiler automatically infers dependencies from resource access patterns.
//! When pass A writes a texture that pass B reads, B is automatically ordered
//! after A. For write-write conflicts without a clear direction, the behavior
//! depends on the [`RenderGraphCompilationMode`].
//!
//! GPU parallelism in this engine happens at the *graph level*, not the *pass level*.
//! The [`FrameSchedule`](crate::scheduler::FrameSchedule) submits multiple graphs
//! with semaphore dependencies, enabling streaming submission and GPU overlap.
//!
//! # Example
//!
//! ```ignore
//! use redlilium_graphics::{GraphicsPass, RenderGraphCompilationMode};
//!
//! let mut graph = schedule.acquire_graph();
//! let geometry = graph.add_graphics_pass(GraphicsPass::new("geometry".into()));
//! let lighting = graph.add_graphics_pass(GraphicsPass::new("lighting".into()));
//! graph.add_dependency(lighting, geometry);
//!
//! let compiled = graph.compile(RenderGraphCompilationMode::Automatic)?;
//! ```

use std::collections::{HashSet, VecDeque};
use std::sync::Arc;

use redlilium_core::pool::Poolable;
use redlilium_core::profiling::{profile_function, profile_scope};

use crate::graph::resource_usage::{PassResourceUsage, SurfaceAccess};
use crate::graph::{Pass, PassHandle, RenderGraph};

/// Controls how the compiler handles ambiguous pass ordering.
///
/// Resource conflicts where the direction is clear (write→read) are always
/// auto-resolved regardless of mode. This setting only affects write-write
/// conflicts where neither pass reads the other's output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RenderGraphCompilationMode {
    /// Auto-resolve ambiguous write-write conflicts by ordering passes
    /// by their addition order (lower index first).
    Automatic,
    /// Return an error if any write-write resource conflicts exist
    /// without explicit ordering between the involved passes.
    #[default]
    Strict,
}

/// A compiled render graph ready for execution.
///
/// Contains a topologically sorted pass order that respects all dependencies,
/// along with pre-computed resource usage for each pass (used for barrier
/// generation by the backend).
#[derive(Debug, Default)]
pub struct CompiledGraph {
    /// Optimized pass execution order as handles.
    pass_order: Vec<PassHandle>,
    /// Pre-computed resource usage per pass, parallel to `pass_order`.
    pass_usages: Vec<PassResourceUsage>,
}

impl PartialEq for CompiledGraph {
    fn eq(&self, other: &Self) -> bool {
        self.pass_order == other.pass_order
    }
}

impl Eq for CompiledGraph {}

impl CompiledGraph {
    /// Create a new compiled graph with the given pass order (test only).
    #[cfg(test)]
    pub(crate) fn new(pass_order: Vec<PassHandle>) -> Self {
        let len = pass_order.len();
        Self {
            pass_order,
            pass_usages: (0..len).map(|_| PassResourceUsage::new()).collect(),
        }
    }

    /// Get the optimized pass execution order as handles.
    pub fn pass_order(&self) -> &[PassHandle] {
        &self.pass_order
    }

    /// Get pre-computed resource usage for each pass.
    ///
    /// Indexed parallel to [`pass_order`](Self::pass_order): `pass_usages()[i]`
    /// corresponds to the pass at `pass_order()[i]`.
    pub fn pass_usages(&self) -> &[PassResourceUsage] {
        &self.pass_usages
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
    fn new_empty() -> Self {
        Self::default()
    }
    fn reset(&mut self) {
        self.pass_order.clear();
        self.pass_usages.clear();
    }
}

/// Errors that can occur during graph compilation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphError {
    /// The graph contains a cyclic dependency.
    ///
    /// This can occur from explicit edges or from contradictory resource
    /// access patterns (e.g., pass A reads what B writes, and B reads what A writes).
    CyclicDependency,

    /// An invalid pass handle was encountered.
    InvalidPassHandle(PassHandle),

    /// Two passes write the same resource but have no explicit ordering.
    ///
    /// Only returned in [`RenderGraphCompilationMode::Strict`] mode.
    AmbiguousOrder {
        pass_a: PassHandle,
        pass_b: PassHandle,
    },
}

impl std::fmt::Display for GraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CyclicDependency => write!(f, "render graph contains cyclic dependency"),
            Self::InvalidPassHandle(handle) => {
                write!(f, "invalid pass handle: {:?}", handle)
            }
            Self::AmbiguousOrder { pass_a, pass_b } => {
                write!(
                    f,
                    "ambiguous ordering between passes {:?} and {:?}: \
                     both write the same resource without explicit dependency",
                    pass_a, pass_b
                )
            }
        }
    }
}

impl std::error::Error for GraphError {}

/// Compile a render graph into an execution plan.
///
/// Performs resource analysis, auto-dependency inference, and topological
/// sorting to produce a pass order that respects all dependencies.
///
/// # Arguments
///
/// * `graph` - The render graph to compile
/// * `mode` - How to handle ambiguous write-write conflicts
pub fn compile(
    graph: &RenderGraph,
    mode: RenderGraphCompilationMode,
) -> Result<CompiledGraph, GraphError> {
    let mut result = CompiledGraph::default();
    compile_into(graph.passes(), graph.edges(), mode, &mut result)?;
    Ok(result)
}

/// Compile a render graph into an existing [`CompiledGraph`], reusing its allocation.
///
/// This is the in-place variant of [`compile`]. It clears the target and fills
/// it with the topologically sorted pass order and pre-computed resource usages.
pub(crate) fn compile_into(
    passes: &[Pass],
    edges: &[(PassHandle, PassHandle)],
    mode: RenderGraphCompilationMode,
    target: &mut CompiledGraph,
) -> Result<(), GraphError> {
    profile_function!();

    let n = passes.len();
    target.pass_order.clear();
    target.pass_usages.clear();

    if n == 0 {
        return Ok(());
    }

    // Step 1: Infer resource usage for each pass
    let mut usages: Vec<PassResourceUsage>;
    {
        profile_scope!("infer_resource_usage");
        usages = passes.iter().map(|p| p.infer_resource_usage()).collect();
    }

    // Step 2: Auto-generate dependency edges from resource access patterns
    let auto_edges;
    {
        profile_scope!("infer_resource_edges");
        auto_edges = infer_resource_edges(&usages, edges, mode)?;
    }

    // Step 3: Merge explicit + auto edges (deduplicated)
    let all_edges: Vec<(PassHandle, PassHandle)>;
    {
        profile_scope!("merge_edges");
        let edge_set: HashSet<(PassHandle, PassHandle)> =
            edges.iter().chain(auto_edges.iter()).copied().collect();
        all_edges = edge_set.into_iter().collect();
    }

    // Step 4: Kahn's algorithm for topological sort
    {
        profile_scope!("compute_in_degree");
        let mut in_degree = vec![0u32; n];
        for &(dependent, _dependency) in &all_edges {
            in_degree[dependent.index()] += 1;
        }

        let mut queue: VecDeque<PassHandle> = (0..n as u32)
            .map(PassHandle::new)
            .filter(|&h| in_degree[h.index()] == 0)
            .collect();

        {
            profile_scope!("topological_sort");
            while let Some(handle) = queue.pop_front() {
                target.pass_order.push(handle);

                for &(dependent, dependency) in &all_edges {
                    if dependency == handle {
                        in_degree[dependent.index()] -= 1;
                        if in_degree[dependent.index()] == 0 {
                            queue.push_back(dependent);
                        }
                    }
                }
            }
        }

        if target.pass_order.len() != n {
            target.pass_order.clear();
            return Err(GraphError::CyclicDependency);
        }
    }

    // Step 5: Store resource usages ordered by pass_order
    {
        profile_scope!("store_usages");
        for &handle in &target.pass_order {
            target
                .pass_usages
                .push(std::mem::take(&mut usages[handle.index()]));
        }
    }

    Ok(())
}

/// Analyze resource conflicts between two passes.
///
/// Returns `(a_before_b, b_before_a, has_waw)`:
/// - `a_before_b`: pass a writes a resource that pass b reads
/// - `b_before_a`: pass b writes a resource that pass a reads
/// - `has_waw`: both passes write the same resource (write-after-write)
fn analyze_resource_conflict(a: &PassResourceUsage, b: &PassResourceUsage) -> (bool, bool, bool) {
    let mut a_before_b = false;
    let mut b_before_a = false;
    let mut has_waw = false;

    // Check texture conflicts
    for ta in &a.texture_usages {
        for tb in &b.texture_usages {
            if Arc::ptr_eq(&ta.texture, &tb.texture) {
                if ta.access.is_write() && tb.access.is_read() {
                    a_before_b = true;
                }
                if tb.access.is_write() && ta.access.is_read() {
                    b_before_a = true;
                }
                if ta.access.is_write() && tb.access.is_write() {
                    has_waw = true;
                }
            }
        }
    }

    // Check buffer conflicts
    for ba in &a.buffer_usages {
        for bb in &b.buffer_usages {
            if Arc::ptr_eq(&ba.buffer, &bb.buffer) {
                if ba.access.is_write() && bb.access.is_read() {
                    a_before_b = true;
                }
                if bb.access.is_write() && ba.access.is_read() {
                    b_before_a = true;
                }
                if ba.access.is_write() && bb.access.is_write() {
                    has_waw = true;
                }
            }
        }
    }

    // Check surface conflicts (swapchain is a single shared resource)
    if let (Some(sa), Some(sb)) = (a.surface_access, b.surface_access) {
        // Both access the surface → always WAW (both variants write)
        has_waw = true;

        // Direction: Write (Clear) → ReadWrite (Load).
        // A pass that loads the surface should come after one that clears it.
        // ReadWrite + ReadWrite is WAW only: both load-modify-store,
        // the ordering is ambiguous without explicit dependency.
        if matches!(sa, SurfaceAccess::Write) && sb.is_read() {
            a_before_b = true;
        }
        if matches!(sb, SurfaceAccess::Write) && sa.is_read() {
            b_before_a = true;
        }
    }

    (a_before_b, b_before_a, has_waw)
}

/// Infer dependency edges from resource access patterns.
///
/// Examines each pair of passes for resource conflicts and generates
/// edges to enforce correct ordering.
fn infer_resource_edges(
    usages: &[PassResourceUsage],
    explicit_edges: &[(PassHandle, PassHandle)],
    mode: RenderGraphCompilationMode,
) -> Result<Vec<(PassHandle, PassHandle)>, GraphError> {
    let n = usages.len();
    if n <= 1 {
        return Ok(Vec::new());
    }

    // Build reachability from explicit edges to detect existing ordering
    let explicit_reach = compute_reachability(n, explicit_edges);

    let mut auto_edges = Vec::new();
    let mut waw_pairs = Vec::new();

    for i in 0..n {
        for j in (i + 1)..n {
            let (i_before_j, j_before_i, has_waw) =
                analyze_resource_conflict(&usages[i], &usages[j]);

            if !i_before_j && !j_before_i && !has_waw {
                continue; // No resource conflict
            }

            let handle_i = PassHandle::new(i as u32);
            let handle_j = PassHandle::new(j as u32);

            // Check if explicit edges already establish ordering
            let explicitly_ordered = explicit_reach[i][j] || explicit_reach[j][i];

            if i_before_j && j_before_i && !explicitly_ordered {
                // Contradictory resource dependencies without explicit resolution
                return Err(GraphError::CyclicDependency);
            }

            if !explicitly_ordered {
                if i_before_j {
                    auto_edges.push((handle_j, handle_i)); // j depends on i
                }
                if j_before_i {
                    auto_edges.push((handle_i, handle_j)); // i depends on j
                }
            }

            // Track WAW pairs that need resolution (no read-based direction)
            if has_waw && !i_before_j && !j_before_i && !explicitly_ordered {
                waw_pairs.push((handle_i, handle_j));
            }
        }
    }

    // Handle write-write conflicts without clear direction
    if !waw_pairs.is_empty() {
        // Recompute reachability including auto edges from write-read analysis
        let combined: Vec<_> = explicit_edges
            .iter()
            .chain(auto_edges.iter())
            .copied()
            .collect();
        let reach = compute_reachability(n, &combined);

        for (a, b) in waw_pairs {
            let ordered = reach[a.index()][b.index()] || reach[b.index()][a.index()];
            if !ordered {
                match mode {
                    RenderGraphCompilationMode::Automatic => {
                        // Lower index pass runs first (preserves addition order)
                        auto_edges.push((b, a));
                    }
                    RenderGraphCompilationMode::Strict => {
                        return Err(GraphError::AmbiguousOrder {
                            pass_a: a,
                            pass_b: b,
                        });
                    }
                }
            }
        }
    }

    Ok(auto_edges)
}

/// Compute transitive reachability matrix using Floyd-Warshall.
///
/// `reach[i][j]` is true if there is a path from pass i to pass j
/// through the given edges. Edge `(dependent, dependency)` means
/// dependency → dependent (dependency can reach dependent).
fn compute_reachability(n: usize, edges: &[(PassHandle, PassHandle)]) -> Vec<Vec<bool>> {
    let mut reach = vec![vec![false; n]; n];

    // Edge (dependent, dependency): dependency runs before dependent
    // So dependency can reach dependent
    for &(dependent, dependency) in edges {
        reach[dependency.index()][dependent.index()] = true;
    }

    for k in 0..n {
        for i in 0..n {
            for j in 0..n {
                if reach[i][k] && reach[k][j] {
                    reach[i][j] = true;
                }
            }
        }
    }

    reach
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{ComputePass, GraphicsPass, TransferPass};

    use RenderGraphCompilationMode::Automatic;

    #[test]
    fn test_compile_empty_graph() {
        let graph = RenderGraph::new();
        let compiled = compile(&graph, Automatic).unwrap();
        assert!(compiled.is_empty());
        assert_eq!(compiled.pass_count(), 0);
        assert!(compiled.pass_usages().is_empty());
    }

    #[test]
    fn test_compile_single_pass() {
        let mut graph = RenderGraph::new();
        let pass = graph.add_graphics_pass(GraphicsPass::new("main".into()));

        let compiled = compile(&graph, Automatic).unwrap();
        assert_eq!(compiled.pass_count(), 1);
        assert_eq!(compiled.pass_order(), &[pass]);
        assert_eq!(compiled.pass_usages().len(), 1);
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

        let compiled = compile(&graph, Automatic).unwrap();
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

        let compiled = compile(&graph, Automatic).unwrap();
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

        let compiled = compile(&graph, Automatic).unwrap();
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

        let result = compile(&graph, Automatic);
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

        let result = compile(&graph, Automatic);
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

        let result = compile(&graph, Automatic);
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

        let compiled = compile(&graph, Automatic).unwrap();
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

        let compiled = compile(&graph, Automatic).unwrap();
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
        assert_eq!(compiled.pass_usages().len(), 3);
    }

    // ========================================================================
    // Auto-dependency tests
    // ========================================================================

    #[test]
    fn test_auto_dependency_write_read_texture() {
        // Pass A writes texture T, Pass B reads texture T
        // → B should automatically depend on A
        use crate::graph::resource_usage::TextureAccessMode;
        use crate::instance::GraphicsInstance;
        use crate::types::{TextureDescriptor, TextureFormat, TextureUsage};

        let instance = GraphicsInstance::new().unwrap();
        let device = instance.create_device().unwrap();
        let texture = device
            .create_texture(&TextureDescriptor::new_2d(
                64,
                64,
                TextureFormat::Rgba8Unorm,
                TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING,
            ))
            .unwrap();

        let usages = vec![
            PassResourceUsage::new()
                .with_texture(texture.clone(), TextureAccessMode::RenderTargetWrite),
            PassResourceUsage::new().with_texture(texture, TextureAccessMode::ShaderRead),
        ];

        let auto = infer_resource_edges(&usages, &[], Automatic).unwrap();
        // Pass 1 (reader) should depend on pass 0 (writer)
        assert!(auto.contains(&(PassHandle::new(1), PassHandle::new(0))));
    }

    #[test]
    fn test_auto_dependency_strict_waw_error() {
        // Both passes write the same texture, no explicit ordering
        // → Strict mode should error
        use crate::graph::resource_usage::TextureAccessMode;
        use crate::instance::GraphicsInstance;
        use crate::types::{TextureDescriptor, TextureFormat, TextureUsage};

        let instance = GraphicsInstance::new().unwrap();
        let device = instance.create_device().unwrap();
        let texture = device
            .create_texture(&TextureDescriptor::new_2d(
                64,
                64,
                TextureFormat::Rgba8Unorm,
                TextureUsage::RENDER_ATTACHMENT,
            ))
            .unwrap();

        let usages = vec![
            PassResourceUsage::new()
                .with_texture(texture.clone(), TextureAccessMode::RenderTargetWrite),
            PassResourceUsage::new().with_texture(texture, TextureAccessMode::RenderTargetWrite),
        ];

        let result = infer_resource_edges(&usages, &[], RenderGraphCompilationMode::Strict);
        assert!(matches!(result, Err(GraphError::AmbiguousOrder { .. })));
    }

    #[test]
    fn test_auto_dependency_automatic_waw() {
        // Both passes write the same texture, no explicit ordering
        // → Automatic mode should pick lower index first
        use crate::graph::resource_usage::TextureAccessMode;
        use crate::instance::GraphicsInstance;
        use crate::types::{TextureDescriptor, TextureFormat, TextureUsage};

        let instance = GraphicsInstance::new().unwrap();
        let device = instance.create_device().unwrap();
        let texture = device
            .create_texture(&TextureDescriptor::new_2d(
                64,
                64,
                TextureFormat::Rgba8Unorm,
                TextureUsage::RENDER_ATTACHMENT,
            ))
            .unwrap();

        let usages = vec![
            PassResourceUsage::new()
                .with_texture(texture.clone(), TextureAccessMode::RenderTargetWrite),
            PassResourceUsage::new().with_texture(texture, TextureAccessMode::RenderTargetWrite),
        ];

        let auto = infer_resource_edges(&usages, &[], Automatic).unwrap();
        // Pass 1 depends on pass 0 (lower index first)
        assert!(auto.contains(&(PassHandle::new(1), PassHandle::new(0))));
    }

    #[test]
    fn test_auto_dependency_explicit_overrides_waw() {
        // Both passes write the same texture, but explicit edge exists
        // → Strict mode should NOT error
        use crate::graph::resource_usage::TextureAccessMode;
        use crate::instance::GraphicsInstance;
        use crate::types::{TextureDescriptor, TextureFormat, TextureUsage};

        let instance = GraphicsInstance::new().unwrap();
        let device = instance.create_device().unwrap();
        let texture = device
            .create_texture(&TextureDescriptor::new_2d(
                64,
                64,
                TextureFormat::Rgba8Unorm,
                TextureUsage::RENDER_ATTACHMENT,
            ))
            .unwrap();

        let usages = vec![
            PassResourceUsage::new()
                .with_texture(texture.clone(), TextureAccessMode::RenderTargetWrite),
            PassResourceUsage::new().with_texture(texture, TextureAccessMode::RenderTargetWrite),
        ];

        let explicit = vec![(PassHandle::new(1), PassHandle::new(0))]; // 1 depends on 0
        let result = infer_resource_edges(&usages, &explicit, RenderGraphCompilationMode::Strict);
        assert!(result.is_ok());
    }

    #[test]
    fn test_auto_dependency_cyclic_resource() {
        // Pass 0 writes T1 (read by pass 1), pass 1 writes T2 (read by pass 0)
        // → Contradictory resource dependencies → CyclicDependency
        use crate::graph::resource_usage::TextureAccessMode;
        use crate::instance::GraphicsInstance;
        use crate::types::{TextureDescriptor, TextureFormat, TextureUsage};

        let instance = GraphicsInstance::new().unwrap();
        let device = instance.create_device().unwrap();
        let tex1 = device
            .create_texture(&TextureDescriptor::new_2d(
                64,
                64,
                TextureFormat::Rgba8Unorm,
                TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING,
            ))
            .unwrap();
        let tex2 = device
            .create_texture(&TextureDescriptor::new_2d(
                64,
                64,
                TextureFormat::Rgba8Unorm,
                TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING,
            ))
            .unwrap();

        let usages = vec![
            PassResourceUsage::new()
                .with_texture(tex1.clone(), TextureAccessMode::RenderTargetWrite)
                .with_texture(tex2.clone(), TextureAccessMode::ShaderRead),
            PassResourceUsage::new()
                .with_texture(tex1, TextureAccessMode::ShaderRead)
                .with_texture(tex2, TextureAccessMode::RenderTargetWrite),
        ];

        let result = infer_resource_edges(&usages, &[], Automatic);
        assert_eq!(result, Err(GraphError::CyclicDependency));
    }

    #[test]
    fn test_auto_dependency_no_conflict_both_read() {
        // Both passes read the same texture → no dependency needed
        use crate::graph::resource_usage::TextureAccessMode;
        use crate::instance::GraphicsInstance;
        use crate::types::{TextureDescriptor, TextureFormat, TextureUsage};

        let instance = GraphicsInstance::new().unwrap();
        let device = instance.create_device().unwrap();
        let texture = device
            .create_texture(&TextureDescriptor::new_2d(
                64,
                64,
                TextureFormat::Rgba8Unorm,
                TextureUsage::TEXTURE_BINDING,
            ))
            .unwrap();

        let usages = vec![
            PassResourceUsage::new().with_texture(texture.clone(), TextureAccessMode::ShaderRead),
            PassResourceUsage::new().with_texture(texture, TextureAccessMode::ShaderRead),
        ];

        let auto = infer_resource_edges(&usages, &[], Automatic).unwrap();
        assert!(auto.is_empty());
    }
}
