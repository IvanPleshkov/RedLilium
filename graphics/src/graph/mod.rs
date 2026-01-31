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
//! # Example
//!
//! ```ignore
//! use redlilium_graphics::{RenderGraph, ColorAttachment, RenderTargetConfig};
//!
//! let mut graph = RenderGraph::new();
//! let pass = graph.add_graphics_pass("main");
//!
//! // Configure render targets
//! pass.as_graphics().unwrap().set_render_targets(
//!     RenderTargetConfig::new()
//!         .with_color(ColorAttachment::from_surface(&surface_texture)
//!             .with_clear_color(0.0, 0.0, 0.0, 1.0))
//! );
//! ```

mod pass;
mod target;
mod transfer;

use std::sync::Arc;

pub use pass::{ComputePass, GraphicsPass, Pass, TransferPass};
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
/// let geometry = graph.add_graphics_pass("geometry");
/// let lighting = graph.add_graphics_pass("lighting");
/// lighting.add_dependency(&geometry);
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
    passes: Vec<Arc<Pass>>,
}

impl RenderGraph {
    /// Create a new empty render graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a graphics pass to the graph.
    ///
    /// Returns an `Arc<Pass>` for direct manipulation and dependency tracking.
    pub fn add_graphics_pass(&mut self, name: impl Into<String>) -> Arc<Pass> {
        let pass = Arc::new(Pass::Graphics(GraphicsPass::new(name.into())));
        self.passes.push(Arc::clone(&pass));
        pass
    }

    /// Add a transfer pass to the graph.
    ///
    /// Returns an `Arc<Pass>` for direct manipulation and dependency tracking.
    pub fn add_transfer_pass(&mut self, name: impl Into<String>) -> Arc<Pass> {
        let pass = Arc::new(Pass::Transfer(TransferPass::new(name.into())));
        self.passes.push(Arc::clone(&pass));
        pass
    }

    /// Add a compute pass to the graph.
    ///
    /// Returns an `Arc<Pass>` for direct manipulation and dependency tracking.
    pub fn add_compute_pass(&mut self, name: impl Into<String>) -> Arc<Pass> {
        let pass = Arc::new(Pass::Compute(ComputePass::new(name.into())));
        self.passes.push(Arc::clone(&pass));
        pass
    }

    /// Get all passes in the graph.
    pub fn passes(&self) -> &[Arc<Pass>] {
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
        Ok(CompiledGraph {
            pass_order: self.passes.clone(),
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
    pass_order: Vec<Arc<Pass>>,
}

impl CompiledGraph {
    /// Get the optimized pass execution order.
    pub fn pass_order(&self) -> &[Arc<Pass>] {
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
        let pass = graph.add_graphics_pass("test_pass");
        assert_eq!(graph.pass_count(), 1);
        assert_eq!(pass.name(), "test_pass");
        assert!(pass.is_graphics());
    }

    #[test]
    fn test_add_transfer_pass() {
        let mut graph = RenderGraph::new();
        let pass = graph.add_transfer_pass("upload");
        assert_eq!(graph.pass_count(), 1);
        assert_eq!(pass.name(), "upload");
        assert!(pass.is_transfer());
    }

    #[test]
    fn test_add_compute_pass() {
        let mut graph = RenderGraph::new();
        let pass = graph.add_compute_pass("simulation");
        assert_eq!(graph.pass_count(), 1);
        assert_eq!(pass.name(), "simulation");
        assert!(pass.is_compute());
    }

    #[test]
    fn test_clear() {
        let mut graph = RenderGraph::new();
        graph.add_graphics_pass("test_pass");

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
        let pass = graph.add_graphics_pass("main");

        let config = RenderTargetConfig::new().with_color(
            ColorAttachment::from_texture(texture).with_clear_color(0.0, 0.0, 0.0, 1.0),
        );

        pass.as_graphics().unwrap().set_render_targets(config);

        assert!(pass.as_graphics().unwrap().has_render_targets());
    }

    #[test]
    fn test_add_dependency() {
        let mut graph = RenderGraph::new();
        let pass1 = graph.add_graphics_pass("geometry");
        let pass2 = graph.add_graphics_pass("lighting");

        pass2.add_dependency(&pass1);

        assert_eq!(pass2.dependencies().len(), 1);
    }

    #[test]
    fn test_compile() {
        let mut graph = RenderGraph::new();
        let pass1 = graph.add_graphics_pass("geometry");
        let pass2 = graph.add_graphics_pass("lighting");
        pass2.add_dependency(&pass1);

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

        let mut graph = RenderGraph::new();
        let transfer = graph.add_transfer_pass("upload");

        let config = TransferConfig::new()
            .with_operation(TransferOperation::copy_buffer_whole(src_buffer, dst_buffer));

        transfer.as_transfer().unwrap().set_transfer_config(config);

        assert!(transfer.as_transfer().unwrap().has_transfers());
        assert!(transfer.is_transfer());
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

        // Transfer pass uploads data
        let upload = graph.add_transfer_pass("upload");
        upload.as_transfer().unwrap().set_transfer_config(
            TransferConfig::new()
                .with_operation(TransferOperation::copy_buffer_whole(staging, vertex)),
        );

        // Render pass depends on transfer completing
        let render = graph.add_graphics_pass("render");
        render.add_dependency(&upload);

        assert_eq!(graph.pass_count(), 2);
        assert_eq!(render.dependencies().len(), 1);
    }

    #[test]
    fn test_mixed_pass_types() {
        let mut graph = RenderGraph::new();

        let upload = graph.add_transfer_pass("upload");
        let compute = graph.add_compute_pass("simulation");
        let render = graph.add_graphics_pass("render");

        compute.add_dependency(&upload);
        render.add_dependency(&compute);

        assert_eq!(graph.pass_count(), 3);
        assert!(upload.is_transfer());
        assert!(compute.is_compute());
        assert!(render.is_graphics());
    }
}
