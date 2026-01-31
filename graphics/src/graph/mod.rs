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
//! # Architecture
//!
//! `RenderGraph` is the graph layer of the rendering architecture:
//!
//! | Layer | Type | Purpose |
//! |-------|------|---------|
//! | Pipeline | [`FramePipeline`](crate::pipeline::FramePipeline) | Multiple frames in flight |
//! | Schedule | [`FrameSchedule`](crate::scheduler::FrameSchedule) | Streaming graph submission |
//! | **Graph** | [`RenderGraph`] | Pass dependencies (this module) |
//! | Pass | [`GraphicsPass`], [`TransferPass`], [`ComputePass`] | Single GPU operation |
//!
//! For the full architecture documentation, see `docs/ARCHITECTURE.md`.
//!
//! # Example
//!
//! ```ignore
//! use redlilium_graphics::{RenderGraph, GraphicsPass, ColorAttachment, RenderTargetConfig};
//!
//! let mut graph = RenderGraph::new();
//!
//! // Create and configure pass before adding
//! let mut pass = GraphicsPass::new("main".into());
//! pass.set_render_targets(
//!     RenderTargetConfig::new()
//!         .with_color(ColorAttachment::from_surface(&surface_texture)
//!             .with_clear_color(0.0, 0.0, 0.0, 1.0))
//! );
//! let handle = graph.add_graphics_pass(pass);
//! ```

mod pass;
mod target;
mod transfer;

pub use pass::{ComputePass, GraphicsPass, Pass, TransferPass};

/// Handle to a pass in the render graph.
///
/// `PassHandle` is `Copy` and cheap to pass around. It is only valid within
/// the `RenderGraph` that created it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PassHandle(u32);

impl PassHandle {
    fn new(index: u32) -> Self {
        Self(index)
    }

    fn index(self) -> usize {
        self.0 as usize
    }
}
pub use target::{
    ColorAttachment, DepthStencilAttachment, LoadOp, RenderTarget, RenderTargetConfig, StoreOp,
};
pub use transfer::{
    BufferCopyRegion, BufferTextureCopyRegion, BufferTextureLayout, TextureCopyLocation,
    TextureCopyRegion, TextureOrigin, TransferConfig, TransferOperation,
};

/// The render graph describes a frame's rendering operations.
///
/// # Construction
///
/// Build a graph by adding passes:
///
/// ```ignore
/// let mut graph = RenderGraph::new();
/// let geometry = graph.add_graphics_pass(GraphicsPass::new("geometry".into()));
/// let lighting = graph.add_graphics_pass(GraphicsPass::new("lighting".into()));
/// graph.add_dependency(lighting, geometry);
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
    /// All passes in the graph (direct storage, no Arc).
    passes: Vec<Pass>,
    /// Dependency edges stored as (dependent, dependency) pairs.
    /// Using edge list avoids per-pass Vec allocations.
    edges: Vec<(PassHandle, PassHandle)>,
}

impl RenderGraph {
    /// Create a new empty render graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a graphics pass to the graph.
    ///
    /// The pass should be fully configured before adding.
    /// Returns a `PassHandle` for referencing this pass.
    pub fn add_graphics_pass(&mut self, pass: GraphicsPass) -> PassHandle {
        let index = self.passes.len() as u32;
        self.passes.push(Pass::Graphics(pass));
        PassHandle::new(index)
    }

    /// Add a transfer pass to the graph.
    ///
    /// The pass should be fully configured before adding.
    /// Returns a `PassHandle` for referencing this pass.
    pub fn add_transfer_pass(&mut self, pass: TransferPass) -> PassHandle {
        let index = self.passes.len() as u32;
        self.passes.push(Pass::Transfer(pass));
        PassHandle::new(index)
    }

    /// Add a compute pass to the graph.
    ///
    /// The pass should be fully configured before adding.
    /// Returns a `PassHandle` for referencing this pass.
    pub fn add_compute_pass(&mut self, pass: ComputePass) -> PassHandle {
        let index = self.passes.len() as u32;
        self.passes.push(Pass::Compute(pass));
        PassHandle::new(index)
    }

    /// Add a dependency between passes.
    ///
    /// The `dependent` pass will execute after the `dependency` pass.
    pub fn add_dependency(&mut self, dependent: PassHandle, dependency: PassHandle) {
        assert!(
            dependent.index() < self.passes.len(),
            "Invalid dependent handle"
        );
        assert!(
            dependency.index() < self.passes.len(),
            "Invalid dependency handle"
        );
        assert!(dependent != dependency, "Pass cannot depend on itself");

        // Check for duplicates
        let exists = self
            .edges
            .iter()
            .any(|&(d, dep)| d == dependent && dep == dependency);
        if !exists {
            self.edges.push((dependent, dependency));
        }
    }

    /// Get dependencies of a pass.
    ///
    /// Returns an iterator over the dependency handles.
    pub fn dependencies(&self, handle: PassHandle) -> impl Iterator<Item = PassHandle> + '_ {
        self.edges
            .iter()
            .filter(move |&&(dependent, _)| dependent == handle)
            .map(|&(_, dependency)| dependency)
    }

    /// Get the number of dependencies for a pass.
    pub fn dependency_count(&self, handle: PassHandle) -> usize {
        self.edges
            .iter()
            .filter(|&&(dependent, _)| dependent == handle)
            .count()
    }

    /// Get all passes in the graph.
    pub fn passes(&self) -> &[Pass] {
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
    pub fn compile(&self) -> Result<CompiledGraph, GraphError> {
        // TODO: Implement proper topological sort
        let pass_order = (0..self.passes.len() as u32).map(PassHandle::new).collect();
        Ok(CompiledGraph { pass_order })
    }

    /// Clear all passes from the graph.
    pub fn clear(&mut self) {
        self.passes.clear();
        self.edges.clear();
    }
}

/// A compiled render graph ready for execution.
#[derive(Debug)]
pub struct CompiledGraph {
    /// Optimized pass execution order as handles.
    pass_order: Vec<PassHandle>,
}

impl CompiledGraph {
    /// Get the optimized pass execution order as handles.
    pub fn pass_order(&self) -> &[PassHandle] {
        &self.pass_order
    }
}

/// Errors that can occur during graph construction or compilation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphError {
    /// The graph contains a cycle.
    CyclicDependency,
}

impl std::fmt::Display for GraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CyclicDependency => write!(f, "render graph contains cyclic dependency"),
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

    #[test]
    fn test_add_graphics_pass() {
        let mut graph = RenderGraph::new();
        let _handle = graph.add_graphics_pass(GraphicsPass::new("test_pass".into()));
        assert_eq!(graph.pass_count(), 1);
        assert_eq!(graph.passes()[0].name(), "test_pass");
        assert!(graph.passes()[0].is_graphics());
    }

    #[test]
    fn test_add_transfer_pass() {
        let mut graph = RenderGraph::new();
        let _handle = graph.add_transfer_pass(TransferPass::new("upload".into()));
        assert_eq!(graph.pass_count(), 1);
        assert_eq!(graph.passes()[0].name(), "upload");
        assert!(graph.passes()[0].is_transfer());
    }

    #[test]
    fn test_add_compute_pass() {
        let mut graph = RenderGraph::new();
        let _handle = graph.add_compute_pass(ComputePass::new("simulation".into()));
        assert_eq!(graph.pass_count(), 1);
        assert_eq!(graph.passes()[0].name(), "simulation");
        assert!(graph.passes()[0].is_compute());
    }

    #[test]
    fn test_clear() {
        let mut graph = RenderGraph::new();
        graph.add_graphics_pass(GraphicsPass::new("test_pass".into()));

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

        let config = RenderTargetConfig::new().with_color(
            ColorAttachment::from_texture(texture).with_clear_color(0.0, 0.0, 0.0, 1.0),
        );

        // Configure pass before adding to graph
        let mut pass = GraphicsPass::new("main".into());
        pass.set_render_targets(config);
        assert!(pass.has_render_targets());

        let mut graph = RenderGraph::new();
        let _handle = graph.add_graphics_pass(pass);

        // Verify it's still configured after adding
        assert!(
            graph.passes()[0]
                .as_graphics()
                .unwrap()
                .has_render_targets()
        );
    }

    #[test]
    fn test_add_dependency() {
        let mut graph = RenderGraph::new();
        let pass1 = graph.add_graphics_pass(GraphicsPass::new("geometry".into()));
        let pass2 = graph.add_graphics_pass(GraphicsPass::new("lighting".into()));

        graph.add_dependency(pass2, pass1);

        assert_eq!(graph.dependency_count(pass2), 1);
        assert_eq!(graph.dependencies(pass2).next(), Some(pass1));
    }

    #[test]
    fn test_compile() {
        let mut graph = RenderGraph::new();
        let pass1 = graph.add_graphics_pass(GraphicsPass::new("geometry".into()));
        let pass2 = graph.add_graphics_pass(GraphicsPass::new("lighting".into()));
        graph.add_dependency(pass2, pass1);

        let compiled = graph.compile().unwrap();
        assert_eq!(compiled.pass_order().len(), 2);
    }

    #[test]
    fn test_transfer_pass() {
        let instance = GraphicsInstance::new().unwrap();
        let device = instance.create_device().unwrap();

        let src_buffer = device
            .create_buffer(&BufferDescriptor::new(1024, BufferUsage::COPY_SRC))
            .unwrap();
        let dst_buffer = device
            .create_buffer(&BufferDescriptor::new(1024, BufferUsage::COPY_DST))
            .unwrap();

        let config = TransferConfig::new()
            .with_operation(TransferOperation::copy_buffer_whole(src_buffer, dst_buffer));

        // Configure pass before adding
        let mut pass = TransferPass::new("upload".into());
        pass.set_transfer_config(config);
        assert!(pass.has_transfers());

        let mut graph = RenderGraph::new();
        let _handle = graph.add_transfer_pass(pass);

        assert!(graph.passes()[0].as_transfer().unwrap().has_transfers());
        assert!(graph.passes()[0].is_transfer());
    }

    #[test]
    fn test_transfer_before_render() {
        let instance = GraphicsInstance::new().unwrap();
        let device = instance.create_device().unwrap();

        let staging = device
            .create_buffer(&BufferDescriptor::new(1024, BufferUsage::COPY_SRC))
            .unwrap();
        let vertex = device
            .create_buffer(&BufferDescriptor::new(
                1024,
                BufferUsage::COPY_DST | BufferUsage::VERTEX,
            ))
            .unwrap();

        let mut graph = RenderGraph::new();

        // Configure and add transfer pass
        let mut upload_pass = TransferPass::new("upload".into());
        upload_pass.set_transfer_config(
            TransferConfig::new()
                .with_operation(TransferOperation::copy_buffer_whole(staging, vertex)),
        );
        let upload = graph.add_transfer_pass(upload_pass);

        // Render pass depends on transfer completing
        let render = graph.add_graphics_pass(GraphicsPass::new("render".into()));
        graph.add_dependency(render, upload);

        assert_eq!(graph.pass_count(), 2);
        assert_eq!(graph.dependency_count(render), 1);
    }

    #[test]
    fn test_mixed_pass_types() {
        let mut graph = RenderGraph::new();

        let upload = graph.add_transfer_pass(TransferPass::new("upload".into()));
        let compute = graph.add_compute_pass(ComputePass::new("simulation".into()));
        let render = graph.add_graphics_pass(GraphicsPass::new("render".into()));

        graph.add_dependency(compute, upload);
        graph.add_dependency(render, compute);

        assert_eq!(graph.pass_count(), 3);
        assert!(graph.passes()[0].is_transfer());
        assert!(graph.passes()[1].is_compute());
        assert!(graph.passes()[2].is_graphics());
    }
}
