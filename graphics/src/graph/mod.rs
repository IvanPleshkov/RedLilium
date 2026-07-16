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
//! use redlilium_graphics::{GraphicsPass, ColorAttachment, RenderTargetConfig};
//!
//! // In a frame loop, use schedule.acquire_graph() to recycle pooled graphs.
//! let mut graph = schedule.acquire_graph();
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
pub mod resource_usage;
mod target;
mod transfer;

pub use pass::{
    AccelerationStructureBuild, AccelerationStructureBuildPass, ComputePass, DispatchCommand,
    DrawCommand, GraphicsPass, IndirectDrawCommand, MeshTasksDrawCommand, Pass, TransferPass,
};

// Re-export compiler types for convenience
pub use crate::compiler::{CompiledGraph, GraphError, RenderGraphCompilationMode, compile};

use redlilium_core::pool::{Poolable, Pooled};

/// Handle to a pass in the render graph.
///
/// `PassHandle` is `Copy` and cheap to pass around. It is only valid within
/// the `RenderGraph` that created it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PassHandle(u32);

impl PassHandle {
    pub(crate) fn new(index: u32) -> Self {
        Self(index)
    }

    pub(crate) fn index(self) -> usize {
        self.0 as usize
    }
}
pub use target::{
    ColorAttachment, DepthStencilAttachment, LoadOp, RenderTarget, RenderTargetConfig, StoreOp,
};
pub use transfer::{
    BufferCopyRegion, BufferTextureCopyRegion, BufferTextureLayout, COPY_BUFFER_ALIGNMENT,
    COPY_BYTES_PER_ROW_ALIGNMENT, ResolvedBufferTextureLayout, TextureCopyLocation,
    TextureCopyRegion, TextureOrigin, TransferConfig, TransferOperation,
    validate_buffer_copy_alignment,
};

/// The render graph describes a frame's rendering operations.
///
/// # Construction
///
/// Build a graph by adding passes:
///
/// ```ignore
/// // In a frame loop, use schedule.acquire_graph() to recycle pooled graphs.
/// let mut graph = schedule.acquire_graph();
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
///
/// # Compiled Graph Caching
///
/// The graph caches its compiled result. Calling `compile()` multiple times
/// without modifying the graph returns the cached result. Any mutation
/// (adding passes, dependencies, or clearing) invalidates the cache.
#[derive(Debug)]
pub struct RenderGraph {
    /// All passes in the graph (direct storage, no Arc).
    passes: Vec<Pass>,
    /// Dependency edges stored as (dependent, dependency) pairs.
    /// Using edge list avoids per-pass Vec allocations.
    edges: Vec<(PassHandle, PassHandle)>,
    /// Cached compiled result. Uses [`Pooled`] to preserve allocation
    /// across invalidations instead of deallocating with `Option<T>`.
    compiled: Pooled<CompiledGraph>,
    /// Queue routing hint (#47 phase 4, #89). See
    /// [`set_queue_preference`](Self::set_queue_preference).
    queue_preference: QueuePreference,
}

/// Which queue a graph asks to run on (#47 phase 4, #89).
///
/// An explicit placement **hint**, not a command: it is honored only when the
/// device exposes the queue AND the graph's passes are legal on it; otherwise
/// the backend walks down the fallback ladder and ultimately runs the graph
/// on the graphics queue — a first-class, fully supported mode. Placement is
/// deliberately explicit while *ordering* stays automatic: cross-queue
/// synchronization is derived from resource usage by the trackers, never
/// declared by hand.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QueuePreference {
    /// The graphics queue — the default; always honored.
    #[default]
    Graphics,
    /// The async compute queue. Requires a graph with no graphics passes.
    AsyncCompute,
    /// The dedicated transfer queue (DMA engines — host↔device streaming).
    /// Requires a transfer-only graph. Falls back to the async compute
    /// queue, then graphics.
    Transfer,
}

impl RenderGraph {
    /// Create a new empty render graph.
    ///
    /// Prefer [`FrameSchedule::acquire_graph`](crate::scheduler::FrameSchedule::acquire_graph)
    /// in frame loops to reuse pooled graphs and avoid memory leaks.
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            queue_preference: QueuePreference::Graphics,
            passes: Vec::new(),
            edges: Vec::new(),
            compiled: Pooled::default(),
        }
    }

    /// Add a graphics pass to the graph.
    ///
    /// The pass should be fully configured before adding.
    /// Returns a `PassHandle` for referencing this pass.
    ///
    /// Note: Adding a pass invalidates any cached compiled graph.
    pub fn add_graphics_pass(&mut self, pass: GraphicsPass) -> PassHandle {
        self.compiled.release(); // Invalidate cache
        let index = self.passes.len() as u32;
        self.passes.push(Pass::Graphics(pass));
        PassHandle::new(index)
    }

    /// Add a transfer pass to the graph.
    ///
    /// The pass should be fully configured before adding.
    /// Returns a `PassHandle` for referencing this pass.
    ///
    /// Note: Adding a pass invalidates any cached compiled graph.
    pub fn add_transfer_pass(&mut self, pass: TransferPass) -> PassHandle {
        self.compiled.release(); // Invalidate cache
        let index = self.passes.len() as u32;
        self.passes.push(Pass::Transfer(pass));
        PassHandle::new(index)
    }

    /// Add a compute pass to the graph.
    ///
    /// The pass should be fully configured before adding.
    /// Returns a `PassHandle` for referencing this pass.
    ///
    /// Note: Adding a pass invalidates any cached compiled graph.
    pub fn add_compute_pass(&mut self, pass: ComputePass) -> PassHandle {
        self.compiled.release(); // Invalidate cache
        let index = self.passes.len() as u32;
        self.passes.push(Pass::Compute(pass));
        PassHandle::new(index)
    }

    /// Add an acceleration-structure build pass to the graph (#110).
    ///
    /// The pass should be fully configured before adding.
    /// Returns a `PassHandle` for referencing this pass.
    ///
    /// Note: Adding a pass invalidates any cached compiled graph.
    pub fn add_acceleration_structure_build_pass(
        &mut self,
        pass: AccelerationStructureBuildPass,
    ) -> PassHandle {
        self.compiled.release(); // Invalidate cache
        let index = self.passes.len() as u32;
        self.passes.push(Pass::AccelerationStructureBuild(pass));
        PassHandle::new(index)
    }

    /// Drive transparent BLAS compaction (#110 phase 3, ADR-032).
    ///
    /// For every BLAS created with
    /// [`BlasDescriptor::with_compaction`](crate::BlasDescriptor::with_compaction)
    /// whose GPU-side compacted size has come back (a few frames after its
    /// build), this adds a compaction-copy pass to the graph; the structure is
    /// swapped for the smaller copy once that copy is guaranteed complete, with
    /// no caller-visible change. Call once per frame after building the frame's
    /// other passes, like an upload flush — a no-op when nothing is ready.
    pub fn flush_acceleration_structure_compaction(
        &mut self,
        device: &std::sync::Arc<crate::device::GraphicsDevice>,
    ) {
        device.flush_blas_compaction(self);
    }

    /// Add a dependency between passes.
    ///
    /// The `dependent` pass will execute after the `dependency` pass.
    ///
    /// Note: Adding a dependency invalidates any cached compiled graph.
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
            self.compiled.release(); // Invalidate cache
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

    /// Whether the graph contains any graphics (raster) pass.
    ///
    /// Graphs with graphics passes can only run on the graphics queue;
    /// compute/transfer-only graphs are eligible for async compute routing.
    pub fn has_graphics_passes(&self) -> bool {
        self.passes.iter().any(|pass| pass.as_graphics().is_some())
    }

    /// Request routing this graph to a specific queue (#47 phase 4, #89).
    ///
    /// See [`QueuePreference`] for the honoring rules and fallback ladder.
    pub fn set_queue_preference(&mut self, preference: QueuePreference) {
        self.queue_preference = preference;
    }

    /// Which queue this graph requests.
    pub fn queue_preference(&self) -> QueuePreference {
        self.queue_preference
    }

    /// Whether every pass in the graph is a transfer pass (the only pass
    /// kind legal on a transfer-only queue family).
    pub fn is_transfer_only(&self) -> bool {
        self.passes
            .iter()
            .all(|pass| matches!(pass, Pass::Transfer(_)))
    }

    /// Whether the graph contains an operation that is legal only on a
    /// graphics-capable queue (#96): a transfer pass with a
    /// [`GenerateMipmaps`](crate::graph::TransferOperation::GenerateMipmaps) op
    /// (`vkCmdBlitImage`), which the transfer and async-compute families cannot
    /// execute. Such a graph must run on the graphics queue.
    pub fn requires_graphics_queue(&self) -> bool {
        self.passes.iter().any(|pass| match pass {
            Pass::Transfer(p) => p.requires_graphics_queue(),
            _ => false,
        })
    }

    /// Whether any pass renders to the swapchain (surface) image.
    ///
    /// At most one such graph may be submitted per frame: its submit is
    /// synchronized against the per-frame acquire/present semaphore pair
    /// (enforced in [`FrameSchedule::submit`](crate::scheduler::FrameSchedule::submit)).
    pub fn writes_swapchain(&self) -> bool {
        self.passes.iter().any(|pass| {
            pass.as_graphics()
                .and_then(|g| g.render_targets())
                .is_some_and(|targets| {
                    targets
                        .color_attachments
                        .iter()
                        .any(|attachment| matches!(attachment.target, RenderTarget::Surface { .. }))
                })
        })
    }

    /// Get all dependency edges in the graph.
    ///
    /// Each edge is `(dependent, dependency)` meaning the dependent pass
    /// must execute after the dependency pass.
    pub fn edges(&self) -> &[(PassHandle, PassHandle)] {
        &self.edges
    }

    /// Compile the graph for execution.
    ///
    /// This performs:
    /// - Resource usage inference for each pass
    /// - Auto-dependency generation from resource access patterns
    /// - Topological sorting of passes
    ///
    /// The result is cached; subsequent calls return the cached result
    /// until the graph is modified.
    ///
    /// See [`crate::compiler`] module for implementation details.
    pub fn compile(
        &mut self,
        mode: RenderGraphCompilationMode,
    ) -> Result<&CompiledGraph, GraphError> {
        if !self.compiled.is_active() {
            let target = self.compiled.activate();
            if let Err(e) = crate::compiler::compile_into(&self.passes, &self.edges, mode, target) {
                self.compiled.release();
                return Err(e);
            }
        }
        Ok(self.compiled.get().unwrap())
    }

    /// Get the cached compiled graph, if available.
    ///
    /// Returns `None` if the graph hasn't been compiled yet or if
    /// it has been modified since the last compilation.
    pub fn compiled(&self) -> Option<&CompiledGraph> {
        self.compiled.get()
    }

    /// Invalidate the cached compiled graph.
    ///
    /// The next call to `compile()` will recompute the compilation.
    pub fn invalidate_compiled(&mut self) {
        self.compiled.release();
    }
}

impl Poolable for RenderGraph {
    fn new_empty() -> Self {
        Self::new()
    }
    fn reset(&mut self) {
        self.passes.clear();
        self.edges.clear();
        self.compiled.release();
        self.queue_preference = QueuePreference::Graphics;
    }
}

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

    /// #110 phase 3: an acceleration-structure-build-only graph carries no
    /// graphics passes and is not transfer-only, so an `AsyncCompute`
    /// preference clears the `route_graph` gates and the build runs on the
    /// async compute queue (falling back to graphics on devices without one).
    #[test]
    fn as_build_only_graph_is_async_eligible() {
        let mut graph = RenderGraph::new();
        graph.set_queue_preference(QueuePreference::AsyncCompute);
        graph.add_acceleration_structure_build_pass(AccelerationStructureBuildPass::new(
            "blas build".into(),
        ));
        assert_eq!(graph.queue_preference(), QueuePreference::AsyncCompute);
        // The two gates `route_graph` checks before honoring the async hint.
        assert!(!graph.has_graphics_passes());
        assert!(!graph.is_transfer_only());
        assert!(!graph.requires_graphics_queue());
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
    fn test_reset() {
        let mut graph = RenderGraph::new();
        graph.add_graphics_pass(GraphicsPass::new("test_pass".into()));

        graph.reset();

        assert_eq!(graph.pass_count(), 0);
    }

    /// The bindless heap group (#117) declares every registered texture as a
    /// shader read, so a freshly uploaded texture gets its layout transition
    /// before the first bindless draw.
    #[test]
    fn bindless_heap_declares_live_textures() {
        use std::sync::Arc;

        use crate::graph::resource_usage::TextureAccessMode;
        use crate::materials::{
            BindingEntry, BindingGroupDescriptor, BindingLayout, BoundResource, MaterialDescriptor,
            MaterialInstance, ShaderSource,
        };
        use crate::mesh::{MeshDescriptor, VertexLayout};

        let instance = GraphicsInstance::new().unwrap();
        let device = instance.create_device().unwrap();
        let texture = device
            .create_texture(&TextureDescriptor::new_2d(
                4,
                4,
                TextureFormat::Rgba8Unorm,
                TextureUsage::TEXTURE_BINDING,
            ))
            .unwrap();

        // A heap stand-in built directly (the real one is device-owned and
        // Vulkan-only; inference only walks the slot table).
        let slots = Arc::new(crate::bindless::BindlessSlots::new(4, 4));
        slots.allocate_texture(Arc::clone(&texture)).unwrap();
        let layout = Arc::new(BindingLayout::new().with_bindless_textures(0));
        let mut descriptor = BindingGroupDescriptor::new();
        descriptor.entries.push(BindingEntry::new(
            0,
            BoundResource::BindlessHeap(Arc::clone(&slots)),
        ));
        let group = Arc::new(crate::materials::BindingGroup::new(
            Arc::clone(&device),
            Arc::clone(&layout),
            descriptor,
            crate::backend::GpuBindingGroup::Dummy,
        ));

        let material = Arc::new(crate::materials::Material::new(
            Arc::clone(&device),
            MaterialDescriptor::new()
                .with_shader(ShaderSource::vertex(b"vs".to_vec(), "main"))
                .with_vertex_layout(VertexLayout::position_only())
                .with_binding_layout(layout),
            crate::backend::GpuPipeline::Dummy,
        ));
        let material_instance = Arc::new(MaterialInstance::new(material).with_binding_group(group));
        let mesh = device
            .create_mesh(&MeshDescriptor::new(VertexLayout::position_only()).with_vertex_count(3))
            .unwrap();

        let mut pass = GraphicsPass::new("bindless".into());
        pass.add_draw(mesh, material_instance);

        let usage = pass.infer_resource_usage();
        assert!(
            usage
                .texture_usages
                .iter()
                .any(|d| Arc::ptr_eq(&d.texture, &texture)
                    && d.access == TextureAccessMode::ShaderRead),
            "registered bindless texture must be declared as a shader read"
        );
    }

    /// Mesh-tasks draws (#111) declare the material's storage buffers as pass
    /// inputs (that is their only geometry path — there is no Mesh), so a
    /// GPU write to a meshlet buffer gets a barrier before the draw.
    #[test]
    fn mesh_tasks_draws_declare_material_buffers() {
        use std::sync::Arc;

        use crate::graph::resource_usage::BufferAccessMode;
        use crate::materials::{
            BindingGroupDescriptor, BindingLayout, BindingLayoutEntry, BindingType,
            MaterialDescriptor, MaterialInstance, ShaderSource, ShaderStage,
        };

        let instance = GraphicsInstance::new().unwrap();
        let device = instance.create_device().unwrap();

        let layout = Arc::new(BindingLayout::new().with_entry(BindingLayoutEntry::new(
            0,
            BindingType::StorageBufferReadOnly,
        )));
        let buffer = device
            .create_buffer(&BufferDescriptor::new(256, BufferUsage::STORAGE))
            .unwrap();
        let group = device
            .create_binding_group(
                layout.clone(),
                BindingGroupDescriptor::new().with_buffer(0, buffer.clone()),
            )
            .unwrap();

        // Mesh material built directly around a dummy pipeline handle (no
        // shader compilation — inference only reads the descriptor and the
        // bound groups).
        let descriptor = MaterialDescriptor::new()
            .with_shader(ShaderSource::slang(
                ShaderStage::Mesh,
                b"ms".to_vec(),
                "ms_main",
                vec![],
            ))
            .with_shader(ShaderSource::slang(
                ShaderStage::Fragment,
                b"fs".to_vec(),
                "fs_main",
                vec![],
            ))
            .with_binding_layout(layout)
            .with_label("meshlet test material");
        descriptor.validate_stage_combination().unwrap();
        let material = Arc::new(crate::materials::Material::new(
            device.clone(),
            descriptor,
            crate::backend::GpuPipeline::Dummy,
        ));
        let material_instance = Arc::new(MaterialInstance::new(material).with_binding_group(group));

        let mut pass = GraphicsPass::new("mesh tasks".into());
        assert!(!pass.has_draws());
        pass.add_draw_mesh_tasks(material_instance, [4, 2, 1]);
        assert!(pass.has_draws());
        assert_eq!(pass.mesh_tasks_commands()[0].group_count, [4, 2, 1]);

        let usage = pass.infer_resource_usage();
        assert!(
            usage
                .buffer_usages
                .iter()
                .any(|d| Arc::ptr_eq(&d.buffer, &buffer)
                    && d.access == BufferAccessMode::StorageRead),
            "meshlet storage buffer must be declared as a read"
        );
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

        let compiled = graph
            .compile(RenderGraphCompilationMode::Automatic)
            .unwrap();
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
