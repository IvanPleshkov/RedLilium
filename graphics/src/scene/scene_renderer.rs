//! SceneRenderer manages rendering an ECS world through a render graph.
//!
//! The SceneRenderer is the primary interface for connecting ECS and rendering.
//! It owns a RenderWorld and a RenderGraph, coordinating the extract-prepare-render
//! pipeline each frame.

use crate::backend::{Backend, BackendError};
use crate::graph::{PassType, RenderGraph};

use super::render_world::RenderWorld;

/// SceneRenderer connects an ECS world to a render graph.
///
/// Each ECS world can have its own SceneRenderer with a dedicated render graph.
/// Multiple SceneRenderers share the same backend for efficient GPU resource usage.
///
/// # Lifecycle
///
/// ```ignore
/// let mut renderer = SceneRenderer::new();
///
/// // Each frame:
/// renderer.begin_frame();
/// renderer.extract(&ecs_world);  // Copy ECS data to RenderWorld
/// renderer.prepare();            // Process data for GPU
/// renderer.render(&backend)?;    // Execute render graph
/// renderer.end_frame();
/// ```
#[derive(Debug)]
pub struct SceneRenderer {
    /// Extracted render data from ECS.
    render_world: RenderWorld,
    /// The render graph describing this scene's rendering pipeline.
    render_graph: RenderGraph,
    /// Current frame number.
    frame_count: u64,
    /// Whether we're currently in a frame (between begin_frame and end_frame).
    in_frame: bool,
}

impl Default for SceneRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl SceneRenderer {
    /// Creates a new scene renderer with an empty render graph.
    pub fn new() -> Self {
        Self {
            render_world: RenderWorld::with_capacity(1024, 64, 64),
            render_graph: RenderGraph::new(),
            frame_count: 0,
            in_frame: false,
        }
    }

    /// Returns a reference to the render world.
    #[inline]
    pub fn render_world(&self) -> &RenderWorld {
        &self.render_world
    }

    /// Returns a mutable reference to the render world.
    #[inline]
    pub fn render_world_mut(&mut self) -> &mut RenderWorld {
        &mut self.render_world
    }

    /// Returns a reference to the render graph.
    #[inline]
    pub fn render_graph(&self) -> &RenderGraph {
        &self.render_graph
    }

    /// Returns a mutable reference to the render graph for configuration.
    #[inline]
    pub fn render_graph_mut(&mut self) -> &mut RenderGraph {
        &mut self.render_graph
    }

    /// Returns the current frame count.
    #[inline]
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Begins a new frame.
    ///
    /// This clears the render world and prepares for extraction.
    /// Must be called before extract().
    pub fn begin_frame(&mut self) {
        debug_assert!(!self.in_frame, "begin_frame called while already in frame");
        self.in_frame = true;
        self.render_world.clear();
        self.render_graph.clear();
    }

    /// Prepares extracted data for GPU upload.
    ///
    /// This phase processes the RenderWorld data into GPU-ready formats:
    /// - Sorts items for optimal rendering order
    /// - Batches items by material/mesh
    /// - Prepares instance data buffers
    ///
    /// For now, this is a placeholder - actual GPU preparation will be added
    /// when we implement real rendering.
    pub fn prepare(&mut self) {
        debug_assert!(self.in_frame, "prepare called outside of frame");

        // Sorting will use camera position when camera is implemented
        // For now, just log item counts
        log::trace!(
            "Prepare phase: {} opaque, {} masked, {} transparent items",
            self.render_world.opaque_items().len(),
            self.render_world.masked_items().len(),
            self.render_world.transparent_items().len()
        );
    }

    /// Sets up a basic render graph for the scene.
    ///
    /// This creates a simple forward rendering pipeline:
    /// - Geometry pass: render all opaque/masked items
    /// - Transparent pass: render transparent items
    pub fn setup_basic_graph(&mut self) {
        debug_assert!(self.in_frame, "setup_basic_graph called outside of frame");

        // Add a geometry pass for opaque objects
        let _geometry_pass = self.render_graph.add_pass("geometry", PassType::Graphics);

        // Add a transparent pass
        let _transparent_pass = self
            .render_graph
            .add_pass("transparent", PassType::Graphics);
    }

    /// Executes the render graph with the given backend.
    ///
    /// This runs the prepared render graph, issuing draw calls for all
    /// items in the render world.
    pub fn render<B: Backend>(&mut self, backend: &B) -> Result<(), BackendError> {
        debug_assert!(self.in_frame, "render called outside of frame");

        // Compile the render graph
        let compiled = self.render_graph.compile().map_err(|e| {
            log::error!("Failed to compile render graph: {}", e);
            BackendError::Internal(format!("Graph compilation failed: {}", e))
        })?;

        // Execute the compiled graph
        backend.execute_graph(&compiled)?;

        log::trace!(
            "Rendered frame {} with {} items",
            self.frame_count,
            self.render_world.total_items()
        );

        Ok(())
    }

    /// Ends the current frame.
    ///
    /// Must be called after render() to finalize the frame.
    pub fn end_frame(&mut self) {
        debug_assert!(self.in_frame, "end_frame called outside of frame");
        self.in_frame = false;
        self.frame_count = self.frame_count.wrapping_add(1);
    }

    /// Convenience method to run a complete frame.
    ///
    /// Equivalent to calling begin_frame, prepare, setup_basic_graph, render, end_frame.
    pub fn render_frame<B: Backend>(&mut self, backend: &B) -> Result<(), BackendError> {
        if !self.in_frame {
            self.begin_frame();
        }

        self.prepare();
        self.setup_basic_graph();
        self.render(backend)?;
        self.end_frame();

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::DummyBackend;

    #[test]
    fn scene_renderer_lifecycle() {
        let mut renderer = SceneRenderer::new();
        let backend = DummyBackend::new();

        renderer.begin_frame();
        renderer.prepare();
        renderer.setup_basic_graph();
        assert!(renderer.render(&backend).is_ok());
        renderer.end_frame();

        assert_eq!(renderer.frame_count(), 1);
    }

    #[test]
    fn scene_renderer_render_frame() {
        let mut renderer = SceneRenderer::new();
        let backend = DummyBackend::new();

        assert!(renderer.render_frame(&backend).is_ok());
        assert_eq!(renderer.frame_count(), 1);

        assert!(renderer.render_frame(&backend).is_ok());
        assert_eq!(renderer.frame_count(), 2);
    }
}
