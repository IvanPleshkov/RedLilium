//! Render graph executor

use crate::backend::traits::*;
use crate::backend::types::*;
use crate::render_graph::graph::*;
use crate::render_graph::pass::*;
use crate::render_graph::resource::*;
use crate::scene::Scene;
use std::collections::HashMap;

/// Executor for running the compiled render graph
pub struct RenderGraphExecutor {
    /// Allocated textures mapped by resource ID
    allocated_textures: HashMap<ResourceId, TextureHandle>,
    allocated_texture_views: HashMap<ResourceId, TextureViewHandle>,

    /// Allocated buffers mapped by resource ID
    allocated_buffers: HashMap<ResourceId, BufferHandle>,

    /// External texture views (like swapchain)
    external_views: HashMap<ResourceId, TextureViewHandle>,
}

impl RenderGraphExecutor {
    pub fn new() -> Self {
        Self {
            allocated_textures: HashMap::new(),
            allocated_texture_views: HashMap::new(),
            allocated_buffers: HashMap::new(),
            external_views: HashMap::new(),
        }
    }

    /// Set an external texture view (e.g., swapchain image)
    pub fn set_external_view(&mut self, resource: ResourceId, view: TextureViewHandle) {
        self.external_views.insert(resource, view);
    }

    /// Allocate resources needed for the render graph
    pub fn allocate_resources<B: GraphicsBackend>(
        &mut self,
        graph: &RenderGraph,
        compiled: &CompiledGraph,
        backend: &mut B,
    ) -> BackendResult<()> {
        // Allocate textures
        for resource in graph.resources() {
            match resource {
                VirtualResource::Texture(tex) => {
                    if !self.allocated_textures.contains_key(&tex.id) {
                        let handle = backend.create_texture(&tex.desc)?;
                        let view = backend.create_texture_view(handle)?;
                        self.allocated_textures.insert(tex.id, handle);
                        self.allocated_texture_views.insert(tex.id, view);
                    }
                }
                VirtualResource::Buffer(buf) => {
                    if !self.allocated_buffers.contains_key(&buf.id) {
                        let handle = backend.create_buffer(&buf.desc)?;
                        self.allocated_buffers.insert(buf.id, handle);
                    }
                }
                VirtualResource::External(_) => {
                    // External resources are set via set_external_view
                }
            }
        }

        Ok(())
    }

    /// Execute the render graph
    pub fn execute<B: GraphicsBackend + 'static>(
        &self,
        graph: &RenderGraph,
        compiled: &CompiledGraph,
        backend: &mut B,
        scene: &Scene,
        width: u32,
        height: u32,
    ) {
        // Build resource maps
        let mut texture_views: HashMap<ResourceId, TextureViewHandle> = HashMap::new();

        // Add allocated texture views
        texture_views.extend(self.allocated_texture_views.iter().map(|(&k, &v)| (k, v)));

        // Add external views
        texture_views.extend(self.external_views.iter().map(|(&k, &v)| (k, v)));

        // Execute passes in order
        for &pass_id in &compiled.pass_order {
            if let Some(pass) = graph.get_pass(pass_id) {
                let mut ctx = PassExecuteContext {
                    backend: backend as &mut dyn std::any::Any,
                    scene,
                    width,
                    height,
                    resource_textures: &texture_views,
                    resource_buffers: &self.allocated_buffers,
                };

                pass.execute(&mut ctx);
            }
        }
    }

    /// Clean up allocated resources
    pub fn cleanup<B: GraphicsBackend>(&mut self, backend: &mut B) {
        for (_, handle) in self.allocated_textures.drain() {
            backend.destroy_texture(handle);
        }
        self.allocated_texture_views.clear();

        for (_, handle) in self.allocated_buffers.drain() {
            backend.destroy_buffer(handle);
        }

        self.external_views.clear();
    }
}

impl Default for RenderGraphExecutor {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for creating render graphs with a fluent API
pub struct RenderGraphBuilder {
    graph: RenderGraph,
    screen_width: u32,
    screen_height: u32,
}

impl RenderGraphBuilder {
    pub fn new(screen_width: u32, screen_height: u32) -> Self {
        Self {
            graph: RenderGraph::new(),
            screen_width,
            screen_height,
        }
    }

    /// Register an external resource
    pub fn external(mut self, name: &str) -> (Self, ResourceId) {
        let id = self.graph.register_external(name);
        (self, id)
    }

    /// Add a graphics pass
    pub fn graphics_pass<P: RenderPass + 'static>(mut self, pass: P) -> (Self, PassId) {
        let id = self.graph.add_pass(pass, PassType::Graphics, self.screen_width, self.screen_height);
        (self, id)
    }

    /// Add a compute pass
    pub fn compute_pass<P: RenderPass + 'static>(mut self, pass: P) -> (Self, PassId) {
        let id = self.graph.add_pass(pass, PassType::Compute, self.screen_width, self.screen_height);
        (self, id)
    }

    /// Build the render graph
    pub fn build(self) -> RenderGraph {
        self.graph
    }
}
