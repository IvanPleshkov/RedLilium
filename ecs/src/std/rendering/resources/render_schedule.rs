//! Frame render-graph resource.

use redlilium_graphics::RenderGraph;

/// Resource holding the current frame's single [`RenderGraph`].
///
/// One render graph is built per frame: the application inserts this resource
/// (with a fresh graph) before running ECS systems, rendering systems append
/// their passes to it, and the application extracts the graph afterwards via
/// [`take`](Self::take) and renders it. Pass ordering and barriers are resolved
/// by the graph compiler, so multiple cameras/passes coexist in one graph.
///
/// Additionally holds the frame's **transfer graphs** (asset-upload streaming,
/// #89): transfer-only graphs pushed by [`AssetGpuFlush`](crate::std::assets::AssetGpuFlush)
/// during the Render schedule, drained by the host and submitted — each as its
/// own queue submit — before the frame graph. On hardware with a dedicated
/// transfer queue the uploads then overlap rendering; cross-queue ordering
/// (upload → first use) is derived automatically by the backend trackers.
pub struct RenderSchedule {
    graph: Option<RenderGraph>,
    transfer_graphs: Vec<RenderGraph>,
}

impl RenderSchedule {
    /// Create a render schedule holding the given frame graph.
    pub fn new(graph: RenderGraph) -> Self {
        Self {
            graph: Some(graph),
            transfer_graphs: Vec::new(),
        }
    }

    /// Create an empty render schedule (no active frame).
    pub fn empty() -> Self {
        Self {
            graph: None,
            transfer_graphs: Vec::new(),
        }
    }

    /// Take the frame graph out, leaving this resource empty.
    pub fn take(&mut self) -> Option<RenderGraph> {
        self.graph.take()
    }

    /// Replace the current frame graph.
    pub fn set(&mut self, graph: RenderGraph) {
        self.graph = Some(graph);
    }

    /// Get a reference to the frame graph, if present.
    pub fn graph(&self) -> Option<&RenderGraph> {
        self.graph.as_ref()
    }

    /// Get a mutable reference to the frame graph, if present.
    ///
    /// Rendering systems append their passes here.
    pub fn graph_mut(&mut self) -> Option<&mut RenderGraph> {
        self.graph.as_mut()
    }

    /// Returns `true` if a frame graph is currently held.
    pub fn is_active(&self) -> bool {
        self.graph.is_some()
    }

    /// Queue a transfer-only graph (asset uploads, #89) for this frame.
    pub fn push_transfer_graph(&mut self, graph: RenderGraph) {
        self.transfer_graphs.push(graph);
    }

    /// Drain the frame's transfer graphs. The host submits each before the
    /// frame graph.
    pub fn take_transfer_graphs(&mut self) -> Vec<RenderGraph> {
        std::mem::take(&mut self.transfer_graphs)
    }
}
